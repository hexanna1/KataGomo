#include "../game/randomopening.h"
#include "../game/gamelogic.h"
#include "../core/rand.h"
#include "../program/playutils.h"
#include "../search/asyncbot.h"
using namespace RandomOpening;

void RandomOpening::initializeBalancedRandomOpening(
  Search* botB,
  Search* botW,
  Board& board,
  BoardHistory& hist,
  Player& nextPlayer,
  Rand& gameRand,
  bool forSelfplay) {

  (void)botB;

  double makeOpeningFairRate = forSelfplay ? 0.98 : 1.0;
  double minAcceptRate = forSelfplay ? 0.005 : 0.001;

  if(gameRand.nextBool(makeOpeningFairRate))  // make game fair
  {
    Loc firstMove = Board::NULL_LOC;
    while(1) {
      firstMove = Board::NULL_LOC;
      for(int attempt = 0; attempt < 1000; attempt++) {
        int firstx = gameRand.nextUInt(board.x_size);
        int firsty = gameRand.nextUInt(board.y_size);
        Loc loc = Location::getLoc(firstx, firsty, board.x_size);
        if(board.colors[loc] == C_EMPTY) {
          firstMove = loc;
          break;
        }
      }
      if(firstMove == Board::NULL_LOC)
        firstMove = PlayUtils::chooseRandomLegalMove(board, hist, C_BLACK, gameRand, Board::PASS_LOC);
      if(firstMove == Board::NULL_LOC)
        return;

      Board boardCopy(board);
      BoardHistory histCopy(hist);
      histCopy.makeBoardMoveAssumeLegal(boardCopy, firstMove, C_BLACK);

      NNResultBuf nnbuf;
      MiscNNInputParams nnInputParams;
      botW->nnEvaluator->evaluate(boardCopy, histCopy, C_WHITE, nnInputParams, nnbuf, false);
      std::shared_ptr<NNOutput> nnOutput = std::move(nnbuf.result);

      double winrate = nnOutput->whiteWinProb;
      double bias = 2 * winrate - 1;
      double dropPow = forSelfplay ? 6.0 : 20.0;
      double acceptRate = pow(1 - bias * bias, dropPow);
      acceptRate = std::max(acceptRate, minAcceptRate);
      if(gameRand.nextBool(acceptRate))
        break;
    }

    hist.makeBoardMoveAssumeLegal(board, firstMove, nextPlayer);
    nextPlayer = getOpp(nextPlayer);
  }


}

void RandomOpening::initializeSpecialOpening(
  Search* botB,
  Search* botW,
  Board& board,
  BoardHistory& hist,
  Player& nextPlayer,
  Rand& gameRand) {
  (void)botB;
  (void)botW;

  double fillProb = gameRand.nextExponential() * 0.01;
  randomFillBoard(board, gameRand, fillProb, fillProb);
  nextPlayer = gameRand.nextBool(0.5) ? C_BLACK : C_WHITE;

  auto rules = hist.rules;
  hist.clear(board, nextPlayer, rules);
}

void RandomOpening::initializeCompletelyRandomOpening(
  Board& board,
  BoardHistory& hist,
  Player& nextPlayer,
  Rand& gameRand,
  double areaPropAvg) {
  double fillProb = gameRand.nextExponential() * areaPropAvg;
  randomFillBoard(board, gameRand, fillProb, fillProb);
  nextPlayer = gameRand.nextBool(0.5) ? C_BLACK : C_WHITE;
  auto rules = hist.rules;
  hist.clear(board, nextPlayer, rules);
}

void RandomOpening::randomFillBoard(Board& board, Rand& gameRand, double bProb, double wProb) {
  if(bProb > 0.5)
    bProb = 0.5;
  if(wProb + bProb > 1)
    wProb = 1 - bProb;

  for(int x = 0; x < board.x_size; x++)
    for(int y = 0; y < board.y_size; y++) {
      Loc loc = Location::getLoc(x, y, board.x_size);
      if (board.colors[loc] == C_EMPTY)
      {
        Color c = C_EMPTY;
        double r = gameRand.nextDouble();
        if(r < bProb)
          c = C_BLACK;
        else if(r < bProb + wProb)
          c = C_WHITE;
        board.setStone(loc, c);
      }
    }
}
