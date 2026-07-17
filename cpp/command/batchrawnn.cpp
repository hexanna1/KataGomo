#include "../core/config_parser.h"
#include "../core/global.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../neuralnet/nneval.h"
#include "../program/setup.h"
#include "../command/commandline.h"
#include "../main.h"

#include "../external/nlohmann_json/json.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>

using namespace std;
using json = nlohmann::json;

namespace {

string humanColumn(int x) {
  string s;
  int value = x + 1;
  while(value > 0) {
    value -= 1;
    s.push_back((char)('a' + value % 26));
    value /= 26;
  }
  reverse(s.begin(),s.end());
  return s;
}

int parseHumanColumn(const string& s, size_t end) {
  int value = 0;
  for(size_t i = 0; i < end; i++) {
    char c = s[i];
    if(c >= 'A' && c <= 'Z')
      c = (char)(c - 'A' + 'a');
    if(c < 'a' || c > 'z')
      throw StringError("Invalid Hex move: " + s);
    value = value * 26 + (c - 'a' + 1);
  }
  return value - 1;
}

Loc parseHumanMove(const string& token, const Board& board) {
  size_t split = 0;
  while(split < token.size() && ((token[split] >= 'a' && token[split] <= 'z') || (token[split] >= 'A' && token[split] <= 'Z')))
    split += 1;
  if(split == 0 || split >= token.size())
    throw StringError("Invalid Hex move: " + token);
  int x = parseHumanColumn(token,split);
  int row;
  if(!Global::tryStringToInt(token.substr(split),row))
    throw StringError("Invalid Hex move: " + token);
  int y = row - 1;
  if(x < 0 || x >= board.x_size || y < 0 || y >= board.y_size)
    throw StringError("Hex move out of bounds: " + token);
  return Location::getLoc(x,y,board.x_size);
}


string humanMove(Loc loc, const Board& board) {
  if(loc == Board::PASS_LOC)
    return "pass";
  int x = Location::getX(loc,board.x_size);
  int y = Location::getY(loc,board.x_size);
  return humanColumn(x) + Global::intToString(y + 1);
}

void parsePosition(
  const string& position, int expectedBoardSize, const Rules& rules,
  Board& board, BoardHistory& hist, Player& nextPla
) {
  string trimmed = Global::trim(position);
  size_t comma = trimmed.find(',');
  string sizeStr = comma == string::npos ? trimmed : trimmed.substr(0,comma);
  int boardSize;
  if(!Global::tryStringToInt(sizeStr,boardSize) || boardSize != expectedBoardSize)
    throw StringError("Position has wrong or invalid board size: " + position);

  board = Board(boardSize,boardSize,BoardShape::Hex);
  nextPla = P_BLACK;
  hist.clear(board,nextPla,rules);
  if(comma == string::npos)
    return;

  const string stream = trimmed.substr(comma+1);
  size_t i = 0;
  while(i < stream.size()) {
    Loc loc;
    if(stream.compare(i,2,":p") == 0) {
      loc = Board::PASS_LOC;
      i += 2;
    }
    else {
      size_t start = i;
      while(i < stream.size() && ((stream[i] >= 'a' && stream[i] <= 'z') || (stream[i] >= 'A' && stream[i] <= 'Z')))
        i += 1;
      while(i < stream.size() && stream[i] >= '0' && stream[i] <= '9')
        i += 1;
      if(i == start)
        throw StringError("Invalid position stream: " + position);
      loc = parseHumanMove(stream.substr(start,i-start),board);
    }
    if(!hist.makeBoardMoveTolerant(board,loc,nextPla))
      throw StringError("Illegal move in position stream: " + position);
    nextPla = getOpp(nextPla);
  }
}


struct PolicyMove {
  string move;
  double prior;
};

static json evaluatePosition(
  const string& position, int boardSize, int topN, const Rules& rules,
  const SearchParams& params, NNEvaluator* nnEval
) {
  Board board;
  BoardHistory hist;
  Player nextPla;
  parsePosition(position,boardSize,rules,board,hist,nextPla);

  MiscNNInputParams nnInputParams;
  nnInputParams.playoutDoublingAdvantage =
    (params.playoutDoublingAdvantagePla == C_EMPTY || params.playoutDoublingAdvantagePla == nextPla) ?
    params.playoutDoublingAdvantage : -params.playoutDoublingAdvantage;
  nnInputParams.symmetry = 0;

  NNResultBuf buf;
  const bool skipCache = true;
  nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache);
  const NNOutput& output = *buf.result;

  vector<PolicyMove> moves;
  moves.reserve((size_t)boardSize * (size_t)boardSize + 1);
  for(int y = 0; y < boardSize; y++) {
    for(int x = 0; x < boardSize; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      if(board.colors[loc] != C_EMPTY)
        continue;
      int pos = NNPos::locToPos(loc,board.x_size,output.nnXLen,output.nnYLen);
      moves.push_back(PolicyMove{humanMove(loc,board),std::max(0.0,(double)output.policyProbs[pos])});
    }
  }
  int passPos = NNPos::locToPos(Board::PASS_LOC,board.x_size,output.nnXLen,output.nnYLen);
  moves.push_back(PolicyMove{"pass",std::max(0.0,(double)output.policyProbs[passPos])});
  sort(moves.begin(),moves.end(),[](const PolicyMove& a, const PolicyMove& b) {
    if(a.prior != b.prior)
      return a.prior > b.prior;
    return a.move < b.move;
  });
  if((int)moves.size() > topN)
    moves.resize(topN);

  json ret;
  ret["position"] = position;
  ret["r"] = (int64_t)llround((1.0 - output.whiteWinProb) * 1000000.0);
  ret["m"] = json::array();
  for(const PolicyMove& move: moves)
    ret["m"].push_back({move.move,(int64_t)llround(move.prior * 1000000.0)});
  return ret;
}

}

int MainCmds::batchrawnn(const vector<string>& args) {
  Board::initHash();
  Rand seedRand;

  ConfigParser cfg;
  string modelFile;
  int boardSize;
  int numWorkers;
  int topN;
  bool streamMode;
  try {
    KataGoCommandLine cmd("Evaluate newline-delimited canonical Hex positions using native neural-net batching.");
    cmd.addConfigFileArg("","gtp_example.cfg");
    cmd.addModelFileArg();
    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();
    TCLAP::ValueArg<int> boardSizeArg("","board-size","Hex board size",true,0,"SIZE");
    TCLAP::ValueArg<int> workersArg("","workers","Number of concurrent evaluator calls; defaults to numSearchThreads",false,0,"THREADS");
    TCLAP::ValueArg<int> topNArg("","top-n","Number of policy moves to return",false,24,"MOVES");
    TCLAP::SwitchArg streamArg("","stream","Evaluate blank-line-delimited batches without reloading the model",false);
    cmd.add(boardSizeArg);
    cmd.add(workersArg);
    cmd.add(topNArg);
    cmd.add(streamArg);
    cmd.parseArgs(args);
    modelFile = cmd.getModelFile();
    boardSize = boardSizeArg.getValue();
    numWorkers = workersArg.getValue();
    topN = topNArg.getValue();
    streamMode = streamArg.getValue();
    cmd.getConfig(cfg);
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  if(boardSize < 2 || boardSize > Board::MAX_LEN)
    throw StringError("Invalid board size: " + Global::intToString(boardSize));
  if(topN <= 0 || topN > NNPos::MAX_NN_POLICY_SIZE)
    throw StringError("Invalid top-n: " + Global::intToString(topN));

  Logger logger(nullptr,false,true);
  Rules rules = Setup::loadSingleRules(cfg);
  SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
  if(numWorkers == 0)
    numWorkers = params.numThreads;
  if(numWorkers <= 0 || numWorkers > 16384)
    throw StringError("Invalid number of workers: " + Global::intToString(numWorkers));

  Setup::initializeSession(cfg);
  const int maxConcurrentEvals = numWorkers * 2 + 16;
  const int expectedConcurrentEvals = numWorkers;
  const int defaultMaxBatchSize = std::max(8,((numWorkers+3)/4)*4);
  const bool defaultRequireExactNNLen = true;
  const bool disableFP16 = false;
  NNEvaluator* nnEval = Setup::initializeNNEvaluator(
    modelFile,modelFile,"",cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
    boardSize,boardSize,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
    Setup::SETUP_FOR_GTP
  );
  bool rulesWereSupported;
  rules = nnEval->getSupportedRules(rules,rulesWereSupported);
  if(!rulesWereSupported) {
    delete nnEval;
    NeuralNet::globalCleanup();
    throw StringError("Configured rules are not supported by the neural net");
  }
  atomic<bool> hadError(false);
  auto evaluateBatch = [&](const vector<string>& positions) {
    if(positions.empty())
      return;
    vector<string> outputs(positions.size());
    atomic<size_t> nextIndex(0);
    vector<thread> workers;
    const int batchWorkers = min<int>(numWorkers,positions.size());
    workers.reserve(batchWorkers);
    for(int threadIdx = 0; threadIdx < batchWorkers; threadIdx++) {
      workers.emplace_back([&]() {
        while(true) {
          size_t idx = nextIndex.fetch_add(1,std::memory_order_relaxed);
          if(idx >= positions.size())
            break;
          try {
            outputs[idx] = evaluatePosition(positions[idx],boardSize,topN,rules,params,nnEval).dump();
          }
          catch(const exception& e) {
            hadError.store(true,std::memory_order_relaxed);
            json ret;
            ret["position"] = positions[idx];
            ret["error"] = e.what();
            outputs[idx] = ret.dump();
          }
        }
      });
    }
    for(thread& worker: workers)
      worker.join();
    for(const string& output: outputs)
      cout << output << endl;
    cout.flush();
  };

  vector<string> positions;
  string line;
  while(getline(cin,line)) {
    line = Global::trim(line);
    if(line.length() > 0) {
      positions.push_back(line);
      continue;
    }
    if(streamMode) {
      evaluateBatch(positions);
      positions.clear();
    }
  }
  evaluateBatch(positions);

  delete nnEval;
  NeuralNet::globalCleanup();
  return hadError.load(std::memory_order_relaxed) ? 1 : 0;
}
