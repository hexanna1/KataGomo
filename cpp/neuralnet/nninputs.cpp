#include "../neuralnet/nninputs.h"

using namespace std;

int NNPos::xyToPos(int x, int y, int nnXLen) {
  return y * nnXLen + x;
}
int NNPos::boardLocToPos(Loc loc, const Board& board, int nnXLen, int nnYLen) {
  int x = Location::getX(loc,board.x_size);
  int y = Location::getY(loc,board.x_size);
  if(!board.isOnBoard(loc) || y >= nnYLen)
    return nnXLen * (nnYLen + 1);
  return y * nnXLen + 2*x + (y & 1);
}

static bool isSecondCrosscutLoc(Loc loc, int boardXSize, QuaxVariant variant) {
  if(!QuaxVariantIO::hasDirectionalCrosscuts(variant) || loc < Board::SECOND_CROSSCUT_LOC_BASE || loc >= Board::MAX_ARR_SIZE)
    return false;
  return loc-Board::SECOND_CROSSCUT_LOC_BASE < (boardXSize-1)*(boardXSize-1);
}

static Loc getPhysicalLoc(Loc loc, int boardXSize, QuaxVariant variant) {
  if(!isSecondCrosscutLoc(loc,boardXSize,variant))
    return loc;
  int index = loc-Board::SECOND_CROSSCUT_LOC_BASE;
  return Location::getLoc(index%(boardXSize-1),2*(index/(boardXSize-1))+1,boardXSize);
}

static Loc getSecondCrosscutLoc(Loc diamondLoc, int boardXSize) {
  int x = Location::getX(diamondLoc,boardXSize);
  int row = Location::getY(diamondLoc,boardXSize)/2;
  return Board::SECOND_CROSSCUT_LOC_BASE + row*(boardXSize-1)+x;
}

int NNPos::locToPos(Loc loc, int boardXSize, int boardYSize, QuaxVariant variant, int nnXLen, int nnYLen) {
  if(loc == Board::PASS_LOC)
    return nnXLen * nnYLen;
  else if(loc == Board::NULL_LOC)
    return nnXLen * (nnYLen + 1);
  bool secondCrosscut = isSecondCrosscutLoc(loc,boardXSize,variant);
  Loc physicalLoc = getPhysicalLoc(loc,boardXSize,variant);
  int x = Location::getX(physicalLoc,boardXSize);
  int y = Location::getY(physicalLoc,boardXSize);
  if(x < 0 || x >= boardXSize || y < 0 || y >= boardYSize || ((y & 1) != 0 && x >= boardXSize-1))
    return nnXLen * (nnYLen + 1);
  int nnX;
  if(QuaxVariantIO::hasDirectionalCrosscuts(variant) && (y & 1) != 0) {
    if(secondCrosscut)
      nnX = 2*x;
    else {
      nnX = 2*x+1;
      y -= 1;
    }
  }
  else
    nnX = 2*x + (y & 1);
  if(nnX < 0 || nnX >= nnXLen || y < 0 || y >= nnYLen)
    return nnXLen * (nnYLen + 1);
  return y * nnXLen + nnX;
}

int NNPos::locToPos(Loc loc, const Board& board, int nnXLen, int nnYLen) {
  return locToPos(loc,board.x_size,board.y_size,board.variant,nnXLen,nnYLen);
}

Loc NNPos::posToLoc(int pos, int boardXSize, int boardYSize, QuaxVariant variant, int nnXLen, int nnYLen) {
  if(pos == nnXLen * nnYLen)
    return Board::PASS_LOC;
  int nnX = pos % nnXLen;
  int y = pos / nnXLen;
  if(nnX < 0 || y < 0 || y >= boardYSize || nnX >= 2*boardXSize-1)
    return Board::NULL_LOC;
  if(QuaxVariantIO::hasDirectionalCrosscuts(variant)) {
    if((nnX & 1) == 0 && (y & 1) == 0) {
      int x = nnX/2;
      return Location::getLoc(x,y,boardXSize);
    }
    if((nnX & 1) != 0 && (y & 1) == 0 && y+1 < boardYSize) {
      int x = (nnX-1)/2;
      return Location::getLoc(x,y+1,boardXSize);
    }
    if((nnX & 1) == 0 && (y & 1) != 0 && nnX/2 < boardXSize-1) {
      Loc diamondLoc = Location::getLoc(nnX/2,y,boardXSize);
      return getSecondCrosscutLoc(diamondLoc,boardXSize);
    }
    return Board::NULL_LOC;
  }
  int parity = y & 1;
  if(nnX < parity || ((nnX-parity) & 1) != 0)
    return Board::NULL_LOC;
  int x = (nnX-parity)/2;
  if(x >= boardXSize || (parity != 0 && x >= boardXSize-1))
    return Board::NULL_LOC;
  return Location::getLoc(x,y,boardXSize);
}

Loc NNPos::posToLoc(int pos, const Board& board, int nnXLen, int nnYLen) {
  return posToLoc(pos,board.x_size,board.y_size,board.variant,nnXLen,nnYLen);
}

bool NNPos::isPassPos(int pos, int nnXLen, int nnYLen) {
  return pos == nnXLen * nnYLen;
}

int NNPos::getPolicySize(int nnXLen, int nnYLen) {
  return nnXLen * nnYLen + 1;
}

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

const Hash128 MiscNNInputParams::ZOBRIST_PLAYOUT_DOUBLINGS =
  Hash128(0xa5e6114d380bfc1dULL, 0x4160557f1222f4adULL);
const Hash128 MiscNNInputParams::ZOBRIST_NN_POLICY_TEMP =
  Hash128(0xebcbdfeec6f4334bULL, 0xb85e43ee243b5ad2ULL);

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

double ScoreValue::whiteWinsOfWinner(Player winner, double noResultUtilityForWhite) {
  if(winner == P_WHITE)
    return 1.0;
  else if(winner == P_BLACK)
    return 0.0;

  assert(winner == C_EMPTY);
  return noResultUtilityForWhite;
}

static const double twoOverPi = 0.63661977236758134308;
static const double piOverTwo = 1.57079632679489661923;


NNOutput::NNOutput()
  :noisedPolicyProbs(NULL)
{}
NNOutput::NNOutput(const NNOutput& other) {
  nnHash = other.nnHash;
  whiteWinProb = other.whiteWinProb;
  whiteLossProb = other.whiteLossProb;
  whiteNoResultProb = other.whiteNoResultProb;
  varTimeLeft = other.varTimeLeft;
  shorttermWinlossError = other.shorttermWinlossError;

  nnXLen = other.nnXLen;
  nnYLen = other.nnYLen;

  if(other.noisedPolicyProbs != NULL) {
    noisedPolicyProbs = new float[NNPos::MAX_NN_POLICY_SIZE];
    std::copy(other.noisedPolicyProbs, other.noisedPolicyProbs + NNPos::MAX_NN_POLICY_SIZE, noisedPolicyProbs);
  }
  else
    noisedPolicyProbs = NULL;

  std::copy(other.policyProbs, other.policyProbs+NNPos::MAX_NN_POLICY_SIZE, policyProbs);
}

NNOutput::NNOutput(const vector<shared_ptr<NNOutput>>& others) {
  assert(others.size() < 1000000);
  int len = (int)others.size();
  float floatLen = (float)len;
  assert(len > 0);
  for(int i = 1; i<len; i++) {
    assert(others[i]->nnHash == others[0]->nnHash);
  }
  nnHash = others[0]->nnHash;

  whiteWinProb = 0.0f;
  whiteLossProb = 0.0f;
  whiteNoResultProb = 0.0f;
  varTimeLeft = 0.0f;
  shorttermWinlossError = 0.0f;
  for(int i = 0; i<len; i++) {
    const NNOutput& other = *(others[i]);
    whiteWinProb += other.whiteWinProb;
    whiteLossProb += other.whiteLossProb;
    whiteNoResultProb += other.whiteNoResultProb;
    varTimeLeft += other.varTimeLeft;
    shorttermWinlossError += other.shorttermWinlossError;
  }
  whiteWinProb /= floatLen;
  whiteLossProb /= floatLen;
  whiteNoResultProb /= floatLen;
  varTimeLeft /= floatLen;
  shorttermWinlossError /= floatLen;

  nnXLen = others[0]->nnXLen;
  nnYLen = others[0]->nnYLen;

  noisedPolicyProbs = NULL;

  //For technical correctness in case of impossibly rare hash collisions:
  //Just give up if they don't all match in move legality
  {
    bool mismatch = false;
    std::fill(policyProbs, policyProbs + NNPos::MAX_NN_POLICY_SIZE, 0.0f);
    for(int i = 0; i<len; i++) {
      const NNOutput& other = *(others[i]);
      for(int pos = 0; pos<NNPos::MAX_NN_POLICY_SIZE; pos++) {
        if(i > 0 && (policyProbs[pos] < 0) != (other.policyProbs[pos] < 0))
          mismatch = true;
        policyProbs[pos] += other.policyProbs[pos];
      }
    }
    //In case of mismatch, just take the first one
    //This should basically never happen, only on true hash collisions
    if(mismatch) {
      const NNOutput& other = *(others[0]);
      std::copy(other.policyProbs, other.policyProbs + NNPos::MAX_NN_POLICY_SIZE, policyProbs);
    }
    else {
      for(int pos = 0; pos<NNPos::MAX_NN_POLICY_SIZE; pos++)
        policyProbs[pos] /= floatLen;
    }
  }

}

NNOutput& NNOutput::operator=(const NNOutput& other) {
  if(&other == this)
    return *this;
  nnHash = other.nnHash;
  whiteWinProb = other.whiteWinProb;
  whiteLossProb = other.whiteLossProb;
  whiteNoResultProb = other.whiteNoResultProb;
  varTimeLeft = other.varTimeLeft;
  shorttermWinlossError = other.shorttermWinlossError;

  nnXLen = other.nnXLen;
  nnYLen = other.nnYLen;

  if(noisedPolicyProbs != NULL)
    delete[] noisedPolicyProbs;
  if(other.noisedPolicyProbs != NULL) {
    noisedPolicyProbs = new float[NNPos::MAX_NN_POLICY_SIZE];
    std::copy(other.noisedPolicyProbs, other.noisedPolicyProbs + NNPos::MAX_NN_POLICY_SIZE, noisedPolicyProbs);
  }
  else
    noisedPolicyProbs = NULL;

  std::copy(other.policyProbs, other.policyProbs+NNPos::MAX_NN_POLICY_SIZE, policyProbs);

  return *this;
}


NNOutput::~NNOutput() {
  if(noisedPolicyProbs != NULL) {
    delete[] noisedPolicyProbs;
    noisedPolicyProbs = NULL;
  }
}


void NNOutput::debugPrint(ostream& out, const Board& board) {
  out << "Win " << Global::strprintf("%.2fc",whiteWinProb*100) << endl;
  out << "Loss " << Global::strprintf("%.2fc",whiteLossProb*100) << endl;
  out << "NoResult " << Global::strprintf("%.2fc",whiteNoResultProb*100) << endl;
  out << "VarTimeLeft " << Global::strprintf("%.1f",varTimeLeft) << endl;
  out << "STWinlossError " << Global::strprintf("%.3f",shorttermWinlossError) << endl;

  out << "Policy" << endl;
  int tensorBoardLen = 2 * board.x_size - 1;
  for(int y = 0; y < tensorBoardLen; y++) {
    for(int x = 0; x < tensorBoardLen; x++) {
      int pos = NNPos::xyToPos(x,y,nnXLen);
      float prob = policyProbs[pos];
      if(prob < 0)
        out << "   - ";
      else
        out << Global::strprintf("%4d ", (int)round(prob * 1000));
    }
    out << endl;
  }

}

//-------------------------------------------------------------------------------------------------------------

static void copyWithSymmetry(const float* src, float* dst, int nSize, int hSize, int wSize, int cSize, bool useNHWC, int symmetry, bool reverse) {
  assert(symmetry >= 0 && symmetry < 8);
  bool transpose = (symmetry & 0x4) != 0;
  bool flipX = (symmetry & 0x2) != 0;
  bool flipY = (symmetry & 0x1) != 0;
  assert(!transpose || hSize == wSize);
  int nStride = hSize * wSize * cSize;
  for(int n = 0; n<nSize; n++) {
    for(int y = 0; y<hSize; y++) {
      for(int x = 0; x<wSize; x++) {
        int symX = transpose ? y : x;
        int symY = transpose ? x : y;
        if(flipX)
          symX = wSize - 1 - symX;
        if(flipY)
          symY = hSize - 1 - symY;
        for(int c = 0; c<cSize; c++) {
          int symC = c;
          if(cSize == NNInputs::NUM_FEATURES_SPATIAL_V7 && (flipX != flipY)) {
            if(c == 6) symC = 7;
            else if(c == 7) symC = 6;
          }
          int rawIdx = useNHWC ? n*nStride + (y*wSize+x)*cSize+c : (n*cSize+c)*hSize*wSize + y*wSize+x;
          int symIdx = useNHWC ? n*nStride + (symY*wSize+symX)*cSize+symC : (n*cSize+symC)*hSize*wSize + symY*wSize+symX;
          if(reverse)
            dst[rawIdx] = src[symIdx];
          else
            dst[symIdx] = src[rawIdx];
        }
      }
    }
  }
}


void SymmetryHelpers::copyInputsWithSymmetry(const float* src, float* dst, int nSize, int hSize, int wSize, int cSize, bool useNHWC, int symmetry) {
  copyWithSymmetry(src, dst, nSize, hSize, wSize, cSize, useNHWC, symmetry, false);
}

void SymmetryHelpers::copyOutputsWithSymmetry(
  const float* src, float* dst, int nSize, int hSize, int wSize, int symmetry,
  QuaxVariant variant, int boardXSize, int boardYSize
) {
  copyWithSymmetry(src, dst, nSize, hSize, wSize, 1, false, symmetry, true);
  if(QuaxVariantIO::hasDirectionalCrosscuts(variant)) {
    for(int n = 0; n < nSize; n++) {
      for(int rawPos = 0; rawPos < hSize*wSize; rawPos++) {
        Loc rawLoc = NNPos::posToLoc(rawPos,boardXSize,boardYSize,variant,wSize,hSize);
        if(rawLoc == Board::NULL_LOC)
          continue;
        Loc symLoc = getSymMoveLoc(rawLoc,boardXSize,boardYSize,variant,symmetry);
        int symPos = NNPos::locToPos(symLoc,boardXSize,boardYSize,variant,wSize,hSize);
        int symX = symPos % wSize;
        int symY = symPos / wSize;
        int tensorBoardLen = 2*boardXSize-1;
        if((symmetry & 0x2) != 0)
          symX += wSize-tensorBoardLen;
        if((symmetry & 0x1) != 0)
          symY += hSize-tensorBoardLen;
        symPos = symY*wSize+symX;
        dst[n*hSize*wSize+rawPos] = src[n*hSize*wSize+symPos];
      }
    }
  }
}

int SymmetryHelpers::invert(int symmetry) {
  return symmetry;
}

int SymmetryHelpers::compose(int firstSymmetry, int nextSymmetry) {
  return firstSymmetry ^ nextSymmetry;
}

int SymmetryHelpers::compose(int firstSymmetry, int nextSymmetry, int nextNextSymmetry) {
  return compose(compose(firstSymmetry,nextSymmetry),nextNextSymmetry);
}

Loc SymmetryHelpers::getSymLoc(int x, int y, int xSize, int ySize, int symmetry) {
  assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
  bool flipX = (symmetry & 0x2) != 0;
  bool flipY = (symmetry & 0x1) != 0;
  if(flipX) { x = xSize - 1 - (y & 1) - x; }
  if(flipY) { y = ySize - y - 1; }
  return Location::getLoc(x,y,xSize);
}

Loc SymmetryHelpers::getSymLoc(int x, int y, const Board& board, int symmetry) {
  return getSymLoc(x,y,board.x_size,board.y_size,symmetry);
}

Loc SymmetryHelpers::getSymLoc(Loc loc, const Board& board, int symmetry) {
  if(loc == Board::NULL_LOC || loc == Board::PASS_LOC)
    return loc;
  if(QuaxVariantIO::hasDirectionalCrosscuts(board.variant))
    return getSymMoveLoc(loc,board,symmetry);
  return getSymLoc(Location::getX(loc,board.x_size), Location::getY(loc,board.x_size), board, symmetry);
}

Loc SymmetryHelpers::getSymLoc(Loc loc, int xSize, int ySize, int symmetry) {
  if(loc == Board::NULL_LOC || loc == Board::PASS_LOC)
    return loc;
  return getSymLoc(Location::getX(loc,xSize), Location::getY(loc,xSize), xSize, ySize, symmetry);
}

static void getSymOctagonCoords(int& x, int& row, int size, int symmetry) {
  if((symmetry & 0x4) != 0)
    std::swap(x,row);
  if((symmetry & 0x2) != 0)
    x = size-1-x;
  if((symmetry & 0x1) != 0)
    row = size-1-row;
}

Loc SymmetryHelpers::getSymMoveLoc(Loc loc, int xSize, int ySize, QuaxVariant variant, int symmetry) {
  assert(symmetry >= 0 && symmetry < 8);
  assert(ySize == 2*xSize-1);
  if(loc == Board::NULL_LOC || loc == Board::PASS_LOC)
    return loc;
  bool secondCrosscut = isSecondCrosscutLoc(loc,xSize,variant);
  Loc physicalLoc = getPhysicalLoc(loc,xSize,variant);
  int x = Location::getX(physicalLoc,xSize);
  int y = Location::getY(physicalLoc,xSize);
  if((y & 1) == 0) {
    int row = y/2;
    getSymOctagonCoords(x,row,xSize,symmetry);
    return Location::getLoc(x,2*row,xSize);
  }

  int row = y/2;
  if(variant == QuaxVariant::DoubleCrosscut) {
    int cornersX[4] = {x,x+1,x,x+1};
    int cornersR[4] = {row,row,row+1,row+1};
    for(int i = 0; i < 4; i++)
      getSymOctagonCoords(cornersX[i],cornersR[i],xSize,symmetry);
    int symX = *std::min_element(cornersX,cornersX+4);
    int symRow = *std::min_element(cornersR,cornersR+4);
    return Location::getLoc(symX,2*symRow+1,xSize);
  }

  int ax = secondCrosscut ? x+1 : x;
  int ar = row;
  int bx = secondCrosscut ? x : x+1;
  int br = row+1;
  getSymOctagonCoords(ax,ar,xSize,symmetry);
  getSymOctagonCoords(bx,br,xSize,symmetry);
  int symX = std::min(ax,bx);
  int symRow = std::min(ar,br);
  Loc symDiamond = Location::getLoc(symX,2*symRow+1,xSize);
  bool symSlash = (ax-bx)*(ar-br) < 0;
  return symSlash ? getSecondCrosscutLoc(symDiamond,xSize) : symDiamond;
}

Loc SymmetryHelpers::getSymMoveLoc(Loc loc, const Board& board, int symmetry) {
  return getSymMoveLoc(loc,board.x_size,board.y_size,board.variant,symmetry);
}


Board SymmetryHelpers::getSymBoard(const Board& board, int symmetry) {
  assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
  Board symBoard(board.x_size,board.y_size,board.variant);
  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      if(!board.isOnBoard(loc))
        continue;
      Loc moveLoc = loc;
      if(board.crosscutDirections[loc] == CROSSCUT_SLASH)
        moveLoc = board.getSecondCrosscutLoc(loc);
      Loc symLoc = getSymMoveLoc(moveLoc,board,symmetry);
      bool suc = symBoard.setStone(symLoc,board.colors[loc]);
      assert(suc);
      (void)suc;
    }
  }
  return symBoard;
}

void SymmetryHelpers::markDuplicateMoveLocs(
  const Board& board,
  const BoardHistory& hist,
  const std::vector<int>* onlySymmetries,
  const std::vector<int>& avoidMoves,
  bool* isSymDupLoc,
  std::vector<int>& validSymmetries
) {
  std::fill(isSymDupLoc, isSymDupLoc + Board::MAX_ARR_SIZE, false);
  validSymmetries.clear();
  validSymmetries.reserve(SymmetryHelpers::NUM_SYMMETRIES);
  validSymmetries.push_back(0);

  for(int symmetry = 1; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
    if(onlySymmetries != NULL && !contains(*onlySymmetries,symmetry))
      continue;

    bool isBoardSym = true;
    for(int y = 0; y < board.y_size; y++) {
      for(int x = 0; x < board.x_size; x++) {
        Loc loc = Location::getLoc(x, y, board.x_size);
        if(!board.isOnBoard(loc))
          continue;
        Loc stateLoc = board.crosscutDirections[loc] == CROSSCUT_SLASH ? board.getSecondCrosscutLoc(loc) : loc;
        Loc symMoveLoc = getSymMoveLoc(stateLoc,board,symmetry);
        Loc symLoc = board.getPhysicalLoc(symMoveLoc);
        bool isStoneSym = board.colors[loc] == board.colors[symLoc];
        if(isStoneSym && board.colors[loc] != C_EMPTY && board.isDiamond(loc))
          isStoneSym = board.crosscutDirections[loc] == board.getCrosscutDirectionForMove(symMoveLoc);
        if(!isStoneSym ) {
          isBoardSym = false;
          break;
        }
      }
      if(!isBoardSym)
        break;
    }
    if(isBoardSym)
      validSymmetries.push_back(symmetry);
  }

  auto markSymmetricMoves = [&](Loc moveLoc) {
    if(avoidMoves.size() > 0 && avoidMoves[moveLoc] > 0)
      return;
    for(int symmetry: validSymmetries) {
      if(symmetry == 0)
        continue;
      Loc symLoc = getSymLoc(moveLoc,board,symmetry);
      if(!isSymDupLoc[moveLoc] && moveLoc != symLoc)
        isSymDupLoc[symLoc] = true;
    }
  };

  //Use a deterministic player-dependent representative from each move orbit.
  if(hist.presumedNextMovePla == P_BLACK) {
    for(int x = board.x_size-1; x >= 0; x--) {
      for(int y = 0; y < board.y_size; y++) {
        Loc loc = Location::getLoc(x, y, board.x_size);
        if(!board.isOnBoard(loc))
          continue;
        markSymmetricMoves(loc);
        if(QuaxVariantIO::hasDirectionalCrosscuts(board.variant) && board.isDiamond(loc))
          markSymmetricMoves(board.getSecondCrosscutLoc(loc));
      }
    }
  }
  else {
    for(int x = 0; x < board.x_size; x++) {
      for(int y = board.y_size-1; y >= 0; y--) {
        Loc loc = Location::getLoc(x, y, board.x_size);
        if(!board.isOnBoard(loc))
          continue;
        markSymmetricMoves(loc);
        if(QuaxVariantIO::hasDirectionalCrosscuts(board.variant) && board.isDiamond(loc))
          markSymmetricMoves(board.getSecondCrosscutLoc(loc));
      }
    }
  }
}

//-------------------------------------------------------------------------------------------------------------

static void setRowBin(float* rowBin, int pos, int feature, float value, int posStride, int featureStride) {
  rowBin[pos * posStride + feature * featureStride] = value;
}

//Currently does NOT depend on history (except for marking ko-illegal spots)
Hash128 NNInputs::getHash(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams
) {
  Hash128 hash =
    BoardHistory::getSituationRulesHash(board, hist, nextPlayer);

  //Fold in whether the game is over or not, since this affects how we compute input features
  //but is not a function necessarily of previous hashed values.
  //If the history is in a weird prolonged state, also treat it similarly.
  if(hist.isGameFinished )
    hash ^= Board::ZOBRIST_GAME_IS_OVER;

  //Fold in asymmetric playout indicator
  if(nnInputParams.playoutDoublingAdvantage != 0) {
    int64_t playoutDoublingsDiscretized = (int64_t)(nnInputParams.playoutDoublingAdvantage*256.0f);
    hash.hash0 += Hash::splitMix64((uint64_t)playoutDoublingsDiscretized);
    hash.hash1 += Hash::basicLCong((uint64_t)playoutDoublingsDiscretized);
    hash ^= MiscNNInputParams::ZOBRIST_PLAYOUT_DOUBLINGS;
  }

  //Fold in policy temperature
  if(nnInputParams.nnPolicyTemperature != 1.0f) {
    int64_t nnPolicyTemperatureDiscretized = (int64_t)(nnInputParams.nnPolicyTemperature*2048.0f);
    hash.hash0 ^= Hash::basicLCong2((uint64_t)nnPolicyTemperatureDiscretized);
    hash.hash1 = Hash::splitMix64(hash.hash1 + (uint64_t)nnPolicyTemperatureDiscretized);
    hash.hash0 += hash.hash1;
    hash ^= MiscNNInputParams::ZOBRIST_NN_POLICY_TEMP;
  }

  // Fold in noResultUtilityForWhite
  if(nnInputParams.noResultUtilityForWhite != 0) {
    int64_t noResultUtilityForWhiteDiscretized = (int64_t)(nnInputParams.noResultUtilityForWhite * 2048.0f);
    hash.hash0 ^= Hash::murmurMix((uint64_t)noResultUtilityForWhiteDiscretized);
    hash.hash1 = Hash::rrmxmx(hash.hash1 + (uint64_t)noResultUtilityForWhiteDiscretized);
    hash.hash0 += hash.hash1;
  }

  // Fold in policyLocalFocus
  if(nnInputParams.policyLocalFocusPow != 0) {
    int64_t policyLocalFocusPowDiscretized = (int64_t)(nnInputParams.policyLocalFocusPow * 2048.0f);
    hash.hash0 ^= Hash::basicLCong2(hash.hash1 + (uint64_t)policyLocalFocusPowDiscretized);
    hash.hash1 = Hash::splitMix64(hash.hash0 + (uint64_t)policyLocalFocusPowDiscretized);
    hash.hash0 += hash.hash1;

    int64_t policyLocalFocusDistDiscretized = (int64_t)(nnInputParams.policyLocalFocusDist * 2048.0f);
    hash.hash0 ^= Hash::basicLCong2(hash.hash1 + (uint64_t)policyLocalFocusDistDiscretized);
    hash.hash1 = Hash::splitMix64(hash.hash0 + (uint64_t)policyLocalFocusDistDiscretized);
    hash.hash0 += hash.hash1;
  }

  return hash;
}

//===========================================================================================
//INPUTSVERSION 7
//===========================================================================================


void NNInputs::fillRowV7(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
) {
  assert(nnXLen <= NNPos::MAX_BOARD_LEN);
  assert(nnYLen <= NNPos::MAX_BOARD_LEN);
  assert(2 * board.x_size - 1 <= nnXLen);
  assert(board.y_size <= nnYLen);
  std::fill(rowBin,rowBin+NUM_FEATURES_SPATIAL_V7*nnXLen*nnYLen,false);
  std::fill(rowGlobal,rowGlobal+NUM_FEATURES_GLOBAL_V7,0.0f);

  Player pla = nextPlayer;
  Player opp = getOpp(pla);
  int xSize = board.x_size;
  int ySize = board.y_size;

  int featureStride;
  int posStride;
  if(useNHWC) {
    featureStride = 1;
    posStride = NNInputs::NUM_FEATURES_SPATIAL_V7;
  }
  else {
    featureStride = nnXLen * nnYLen;
    posStride = 1;
  }

  if(QuaxVariantIO::hasDirectionalCrosscuts(board.variant)) {
    int tensorBoardLen = 2*xSize-1;
    for(int y = 0; y < tensorBoardLen; y++) {
      for(int x = 0; x < tensorBoardLen; x++) {
        int pos = NNPos::xyToPos(x,y,nnXLen);
        setRowBin(rowBin,pos,0,1.0f,posStride,featureStride);
        if((x & 1) != (y & 1))
          setRowBin(rowBin,pos,8,1.0f,posStride,featureStride);
      }
    }
  }

  GameLogic::ResultsBeforeNN resultsBeforeNN = nnInputParams.resultsBeforeNN;
  if(!resultsBeforeNN.inited) {
    resultsBeforeNN.init(board, hist, nextPlayer);
  }

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      Loc loc = Location::getLoc(x,y,xSize);

      if(!board.isOnBoard(loc))
        continue;

      int pos = NNPos::boardLocToPos(loc,board,nnXLen,nnYLen);

      if(board.variant == QuaxVariant::DoubleCrosscut)
        setRowBin(rowBin,pos,0,1.0f,posStride,featureStride);

      Color stone = board.colors[loc];

      //Features 1,2 - pla,opp stone
      //Features 3,4,5 - 1,2,3 libs
      if(stone == pla)
        setRowBin(rowBin,pos,1, 1.0f, posStride, featureStride);
      else if(stone == opp)
        setRowBin(rowBin,pos,2, 1.0f, posStride, featureStride);

      //Feature 5 distinguishes interstitial Quax diamonds from octagons.
      if(board.isDiamond(loc))
        setRowBin(rowBin,pos,5, 1.0f, posStride, featureStride);

      CrosscutDirection direction = board.crosscutDirections[loc];
      if(QuaxVariantIO::hasDirectionalCrosscuts(board.variant)) {
        if(direction == CROSSCUT_BACKSLASH)
          setRowBin(rowBin,pos,6,1.0f,posStride,featureStride);
        if(direction == CROSSCUT_SLASH)
          setRowBin(rowBin,pos,7,1.0f,posStride,featureStride);
      }

    }
  }

  //policyLocalFocus
  if(
    nnInputParams.policyLocalFocusPow > 0 && hist.moveHistory.size() >= 1 &&
    board.isOnBoard(board.getPhysicalLoc(hist.moveHistory[hist.moveHistory.size()-1].loc))) {
    Loc lastMove = hist.moveHistory[hist.moveHistory.size() - 1].loc;

    int pos = NNPos::locToPos(lastMove,board,nnXLen,nnYLen);
    setRowBin(rowBin, pos, 3, 1.0f, posStride, featureStride);
    rowGlobal[1] = 1.0;
    rowGlobal[2] = nnInputParams.policyLocalFocusPow * 3.0;
    rowGlobal[3] = 3.0 / nnInputParams.policyLocalFocusDist;
  }

  //The White position is transposed before reaching the net, so the current player's goal is always vertical.
  rowGlobal[0] = 0.0;
  rowGlobal[18] = QuaxVariantIO::hasDirectionalCrosscuts(board.variant) ? 1.0f : 0.0f;
  if(QuaxVariantIO::hasDirectionalCrosscuts(board.variant))
    rowGlobal[14] = nextPlayer == P_BLACK ? -nnInputParams.noResultUtilityForWhite : nnInputParams.noResultUtilityForWhite;

  //Global features.
  //The first 5 of them were set already above to flag which of the past 5 moves were passes.

  //Scoring
  if(hist.rules.scoringRule == Rules::SCORING_AREA) {}
  else
    ASSERT_UNREACHABLE;

  if(hist.rules.maxMoves > 0 && hist.rules.maxMoves < board.playableArea()) {
    rowGlobal[4] = 1.0;
    rowGlobal[14] =
      nextPlayer == P_BLACK ? -nnInputParams.noResultUtilityForWhite : nnInputParams.noResultUtilityForWhite;
    int mm = hist.rules.maxMoves;
    int movecount = board.numStonesOnBoard();
    int area = board.playableArea();
    int remain = mm - movecount;
    if (remain <= 0)
    {
      cout << board;
      cout << hist.rules;
      cout << "remain move count <= 0 in nninput\n";
      remain = 0;
    }
    rowGlobal[5] = double(mm) / double(area);
    rowGlobal[6] = exp(-double(remain) / 1.5);
    rowGlobal[7] = exp(-double(remain) / 5.0);
    rowGlobal[8] = exp(-double(remain) / 15.0);
    rowGlobal[9] = exp(-double(remain) / 50.0);
    rowGlobal[10] = exp(-double(remain) / 150.0);
    rowGlobal[11] = remain % 2;
    rowGlobal[12] = 2.0 * sqrt(double(area - mm) / double(area));

  }

  // Parameter 15 is used because there's actually a discontinuity in how training behavior works when this is
  // nonzero, no matter how slightly.
  if(nnInputParams.playoutDoublingAdvantage != 0) {
    rowGlobal[15] = 1.0;
    rowGlobal[16] = (float)(0.5 * nnInputParams.playoutDoublingAdvantage);
  }
}
