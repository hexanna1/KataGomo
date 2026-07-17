#include "../neuralnet/nninputs.h"
#include "../game/benty.h"

using namespace std;

int NNPos::xyToPos(int x, int y, int nnXLen) {
  return y * nnXLen + x;
}
int NNPos::locToPos(Loc loc, int boardXSize, int nnXLen, int nnYLen) {
  if(loc == Board::PASS_LOC)
    return nnXLen * nnYLen;
  else if(loc == Board::NULL_LOC)
    return nnXLen * (nnYLen + 1);
  return Location::getY(loc,boardXSize) * nnXLen + Location::getX(loc,boardXSize);
}
Loc NNPos::posToLoc(int pos, int boardXSize, int boardYSize, int nnXLen, int nnYLen) {
  if(pos == nnXLen * nnYLen)
    return Board::PASS_LOC;
  int x = pos % nnXLen;
  int y = pos / nnXLen;
  if(x < 0 || x >= boardXSize || y < 0 || y >= boardYSize)
    return Board::NULL_LOC;
  return Location::getLoc(x,y,boardXSize);
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
  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
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

static const int Y_SYM_PERMS[SymmetryHelpers::NUM_SYMMETRIES][3] = {
  {0, 1, 2},
  {1, 2, 0},
  {2, 0, 1},
  {1, 0, 2},
  {0, 2, 1},
  {2, 1, 0},
};

static bool isTrianglePoint(int x, int y, int size) {
  return x >= 0 && y >= 0 && x < size && y < size && x + y < size;
}

static bool isObtuseYPoint(int x, int y, int size) {
  if(x < 0 || y < 0 || x >= size || y >= size || size <= 0 || size % 2 != 1)
    return false;
  int n = (size - 1) / 2;
  int rx = size - 1 - x;
  int ry = size - 1 - y;
  bool inTopLeftCut = x < n && y < n && x + y < n;
  bool inBottomRightCut = rx < n && ry < n && rx + ry < n;
  return !inTopLeftCut && !inBottomRightCut;
}

static int getPermSign(const int perm[3]) {
  int inversions = 0;
  for(int i = 0; i < 3; i++) {
    for(int j = i+1; j < 3; j++) {
      if(perm[i] > perm[j])
        inversions++;
    }
  }
  return inversions % 2 == 0 ? 1 : -1;
}

static void getYSymCoords(int x, int y, int size, int symmetry, int& sx, int& sy) {
  assert(symmetry >= 0 && symmetry < SymmetryHelpers::NUM_SYMMETRIES);
  assert(isTrianglePoint(x, y, size));

  int coords[3] = {x, y, size - 1 - x - y};
  sx = coords[Y_SYM_PERMS[symmetry][0]];
  sy = coords[Y_SYM_PERMS[symmetry][1]];
  assert(isTrianglePoint(sx, sy, size));
}

static void getObtuseYSymCoords(int x, int y, int size, int symmetry, int& sx, int& sy) {
  assert(symmetry >= 0 && symmetry < SymmetryHelpers::NUM_SYMMETRIES);
  assert(isObtuseYPoint(x, y, size));

  int n = (size - 1) / 2;
  int q = x - n;
  int r = y - n;
  int coords[3] = {q, r, -q - r};
  int sign = getPermSign(Y_SYM_PERMS[symmetry]);
  sx = sign * coords[Y_SYM_PERMS[symmetry][0]] + n;
  sy = sign * coords[Y_SYM_PERMS[symmetry][1]] + n;
  assert(isObtuseYPoint(sx, sy, size));
}

static bool isSymmetryPoint(int x, int y, int xSize, int ySize, BoardShape shape) {
  if(xSize != ySize)
    return false;
  switch(shape) {
  case BoardShape::Y:
    return isTrianglePoint(x, y, xSize);
  case BoardShape::ObtuseY:
    return isObtuseYPoint(x, y, xSize);
  case BoardShape::BentY:
    return BentY::isSupportedTensorLen(xSize) &&
      x >= 0 && y >= 0 && x < xSize && y < ySize &&
      BentY::getTopology(xSize).playable[y * xSize + x];
  default:
    ASSERT_UNREACHABLE;
    return false;
  }
}

static void getSymCoords(int x, int y, int xSize, int ySize, BoardShape shape, int symmetry, int& sx, int& sy) {
  assert(xSize == ySize);
  switch(shape) {
  case BoardShape::Y:
    getYSymCoords(x, y, xSize, symmetry, sx, sy);
    return;
  case BoardShape::ObtuseY:
    getObtuseYSymCoords(x, y, xSize, symmetry, sx, sy);
    return;
  case BoardShape::BentY: {
    int symPos = BentY::getTopology(xSize).symPos[y * xSize + x][symmetry];
    sx = symPos % xSize;
    sy = symPos / xSize;
    return;
  }
  default:
    ASSERT_UNREACHABLE;
  }
}

static int getSymmetryForPerm(const int perm[3]) {
  for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
    if(
      Y_SYM_PERMS[symmetry][0] == perm[0] &&
      Y_SYM_PERMS[symmetry][1] == perm[1] &&
      Y_SYM_PERMS[symmetry][2] == perm[2]
    )
      return symmetry;
  }
  ASSERT_UNREACHABLE;
  return 0;
}

static void copyWithSymmetry(const float* src, float* dst, int nSize, int hSize, int wSize, int cSize, bool useNHWC, int symmetry, BoardShape shape, int boardXSize, int boardYSize, bool reverse) {
  assert(symmetry >= 0 && symmetry < SymmetryHelpers::NUM_SYMMETRIES);
  assert(hSize == wSize);
  if(boardXSize <= 0)
    boardXSize = wSize;
  if(boardYSize <= 0)
    boardYSize = hSize;
  assert(boardXSize <= wSize);
  assert(boardYSize <= hSize);
  assert(boardXSize == boardYSize);
  int totalLen = nSize * hSize * wSize * cSize;
  std::fill(dst, dst + totalLen, 0.0f);

  if(useNHWC) {
    for(int n = 0; n<nSize; n++) {
      for(int y = 0; y<boardYSize; y++) {
        for(int x = 0; x<boardXSize; x++) {
          if(!isSymmetryPoint(x, y, boardXSize, boardYSize, shape))
            continue;
          int sx;
          int sy;
          getSymCoords(x, y, boardXSize, boardYSize, shape, symmetry, sx, sy);
          for(int c = 0; c<cSize; c++) {
            if(reverse)
              dst[((n * hSize + y) * wSize + x) * cSize + c] =
                src[((n * hSize + sy) * wSize + sx) * cSize + c];
            else
              dst[((n * hSize + sy) * wSize + sx) * cSize + c] =
                src[((n * hSize + y) * wSize + x) * cSize + c];
          }
        }
      }
    }
  }
  else {
    int spatialSize = hSize * wSize;
    for(int n = 0; n<nSize; n++) {
      for(int c = 0; c<cSize; c++) {
        int ncBase = (n * cSize + c) * spatialSize;
        for(int y = 0; y<boardYSize; y++) {
          for(int x = 0; x<boardXSize; x++) {
            if(!isSymmetryPoint(x, y, boardXSize, boardYSize, shape))
              continue;
            int sx;
            int sy;
            getSymCoords(x, y, boardXSize, boardYSize, shape, symmetry, sx, sy);
            if(reverse)
              dst[ncBase + y * wSize + x] = src[ncBase + sy * wSize + sx];
            else
              dst[ncBase + sy * wSize + sx] = src[ncBase + y * wSize + x];
          }
        }
      }
    }
  }
}


void SymmetryHelpers::copyInputsWithSymmetry(const float* src, float* dst, int nSize, int hSize, int wSize, int cSize, bool useNHWC, int symmetry, BoardShape shape, int boardXSize, int boardYSize) {
  copyWithSymmetry(src, dst, nSize, hSize, wSize, cSize, useNHWC, symmetry, shape, boardXSize, boardYSize, false);
}

void SymmetryHelpers::copyOutputsWithSymmetry(const float* src, float* dst, int nSize, int hSize, int wSize, int symmetry, BoardShape shape, int boardXSize, int boardYSize) {
  copyWithSymmetry(src, dst, nSize, hSize, wSize, 1, false, symmetry, shape, boardXSize, boardYSize, true);
}

int SymmetryHelpers::invert(int symmetry) {
  assert(symmetry >= 0 && symmetry < SymmetryHelpers::NUM_SYMMETRIES);
  int invPerm[3];
  for(int i = 0; i < 3; i++)
    invPerm[Y_SYM_PERMS[symmetry][i]] = i;
  return getSymmetryForPerm(invPerm);
}

int SymmetryHelpers::compose(int firstSymmetry, int nextSymmetry) {
  assert(firstSymmetry >= 0 && firstSymmetry < SymmetryHelpers::NUM_SYMMETRIES);
  assert(nextSymmetry >= 0 && nextSymmetry < SymmetryHelpers::NUM_SYMMETRIES);
  int composedPerm[3];
  for(int i = 0; i < 3; i++)
    composedPerm[i] = Y_SYM_PERMS[firstSymmetry][Y_SYM_PERMS[nextSymmetry][i]];
  return getSymmetryForPerm(composedPerm);
}

int SymmetryHelpers::compose(int firstSymmetry, int nextSymmetry, int nextNextSymmetry) {
  return compose(compose(firstSymmetry,nextSymmetry),nextNextSymmetry);
}

Loc SymmetryHelpers::getSymLoc(int x, int y, int xSize, int ySize, BoardShape shape, int symmetry) {
  assert(xSize == ySize);
  if(!isSymmetryPoint(x, y, xSize, ySize, shape))
    return Location::getLoc(x, y, xSize);

  int sx;
  int sy;
  getSymCoords(x, y, xSize, ySize, shape, symmetry, sx, sy);
  return Location::getLoc(sx, sy, xSize);
}

Loc SymmetryHelpers::getSymLoc(int x, int y, const Board& board, int symmetry) {
  return getSymLoc(x,y,board.x_size,board.y_size,board.shape,symmetry);
}

Loc SymmetryHelpers::getSymLoc(Loc loc, const Board& board, int symmetry) {
  if(loc == Board::NULL_LOC || loc == Board::PASS_LOC)
    return loc;
  return getSymLoc(Location::getX(loc,board.x_size), Location::getY(loc,board.x_size), board, symmetry);
}

Loc SymmetryHelpers::getSymLoc(Loc loc, int xSize, int ySize, int symmetry) {
  if(loc == Board::NULL_LOC || loc == Board::PASS_LOC)
    return loc;
  return getSymLoc(Location::getX(loc,xSize), Location::getY(loc,xSize), xSize, ySize, BoardShape::Y, symmetry);
}

Loc SymmetryHelpers::getSymLoc(int x, int y, int xSize, int ySize, int symmetry) {
  return getSymLoc(x, y, xSize, ySize, BoardShape::Y, symmetry);
}

Loc SymmetryHelpers::getSymLoc(Loc loc, int xSize, int ySize, BoardShape shape, int symmetry) {
  if(loc == Board::NULL_LOC || loc == Board::PASS_LOC)
    return loc;
  return getSymLoc(Location::getX(loc,xSize), Location::getY(loc,xSize), xSize, ySize, shape, symmetry);
}


Board SymmetryHelpers::getSymBoard(const Board& board, int symmetry) {
  assert(board.x_size == board.y_size);
  Board symBoard(board.x_size, board.y_size, board.shape);
  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      if(!board.isOnBoard(loc))
        continue;
      Color color = board.colors[loc];
      if(color != C_BLACK && color != C_WHITE)
        continue;
      Loc symLoc = getSymLoc(x, y, board, symmetry);
      bool suc = symBoard.setStone(symLoc,color);
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


  //If board has different sizes of x and y, we will not search symmetries involved with transpose.
  int symmetrySearchUpperBound = SymmetryHelpers::NUM_SYMMETRIES;

  for(int symmetry = 1; symmetry < symmetrySearchUpperBound; symmetry++) {
    if(onlySymmetries != NULL && !contains(*onlySymmetries,symmetry))
      continue;

    bool isBoardSym = true;
    for(int y = 0; y < board.y_size; y++) {
      for(int x = 0; x < board.x_size; x++) {
        Loc loc = Location::getLoc(x, y, board.x_size);
        if(!board.isOnBoard(loc))
          continue;
        Loc symLoc = getSymLoc(x, y, board,symmetry);
        bool isStoneSym = (board.colors[loc] == board.colors[symLoc]);
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

  //The way we iterate is to achieve https://senseis.xmp.net/?PlayingTheFirstMoveInTheUpperRightCorner%2FDiscussion
  //Reverse the iteration order for white, so that natural openings result in white on the left and black on the right
  //as is common now in SGFs
  if(hist.presumedNextMovePla == P_BLACK) {
    for(int x = board.x_size-1; x >= 0; x--) {
      for(int y = 0; y < board.y_size; y++) {
        Loc loc = Location::getLoc(x, y, board.x_size);
        if(!board.isOnBoard(loc))
          continue;
        if(avoidMoves.size() > 0 && avoidMoves[loc] > 0)
          continue;
        for(int symmetry: validSymmetries) {
          if(symmetry == 0)
            continue;
          Loc symLoc = getSymLoc(x, y, board, symmetry);
          if(!isSymDupLoc[loc] && loc != symLoc)
            isSymDupLoc[symLoc] = true;
        }
      }
    }
  }
  else {
    for(int x = 0; x < board.x_size; x++) {
      for(int y = board.y_size-1; y >= 0; y--) {
        Loc loc = Location::getLoc(x, y, board.x_size);
        if(!board.isOnBoard(loc))
          continue;
        if(avoidMoves.size() > 0 && avoidMoves[loc] > 0)
          continue;
        for(int symmetry: validSymmetries) {
          if(symmetry == 0)
            continue;
          Loc symLoc = getSymLoc(x, y, board, symmetry);
          if(!isSymDupLoc[loc] && loc != symLoc)
            isSymDupLoc[symLoc] = true;
        }
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
  assert(board.x_size <= nnXLen);
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

  GameLogic::ResultsBeforeNN resultsBeforeNN = nnInputParams.resultsBeforeNN;
  if(!resultsBeforeNN.inited) {
    resultsBeforeNN.init(board, hist, nextPlayer);
  }

  vector<int> plaComponent;
  vector<int> oppComponent;
  vector<int> plaComponentSides;
  vector<int> oppComponentSides;
  auto buildComponents = [&](Player componentPla, vector<int>& componentIds, vector<int>& componentSides) {
    componentIds.assign(Board::MAX_ARR_SIZE, -1);
    vector<Loc> stack;
    for(int boardY = 0; boardY < ySize; boardY++) {
      for(int boardX = 0; boardX < xSize; boardX++) {
        Loc start = Location::getLoc(boardX, boardY, xSize);
        if(board.colors[start] != componentPla || componentIds[start] >= 0)
          continue;
        int component = (int)componentSides.size();
        int sides = 0;
        componentIds[start] = component;
        stack.push_back(start);
        while(!stack.empty()) {
          Loc current = stack.back();
          stack.pop_back();
          sides |= board.sideMask(current);
          Loc adjacent[6];
          int numAdjacent = board.getAdjacentLocs(current, adjacent);
          for(int i = 0; i < numAdjacent; i++) {
            Loc next = adjacent[i];
            if(board.colors[next] == componentPla && componentIds[next] < 0) {
              componentIds[next] = component;
              stack.push_back(next);
            }
          }
        }
        componentSides.push_back(sides);
      }
    }
  };
  if(board.shape == BoardShape::BentY) {
    buildComponents(pla, plaComponent, plaComponentSides);
    buildComponents(opp, oppComponent, oppComponentSides);
  }

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      int pos = NNPos::xyToPos(x,y,nnXLen);
      Loc loc = Location::getLoc(x,y,xSize);
      if(!board.isOnBoard(loc))
        continue;

      //Feature 0 - on board
      setRowBin(rowBin,pos,0, 1.0f, posStride, featureStride);

      Color stone = board.colors[loc];

      //Features 1,2 - pla,opp stone
      if(stone == pla)
        setRowBin(rowBin,pos,1, 1.0f, posStride, featureStride);
      else if(stone == opp)
        setRowBin(rowBin,pos,2, 1.0f, posStride, featureStride);

      if(board.shape == BoardShape::BentY) {
        Loc adjacent[6];
        int numAdjacent = board.getAdjacentLocs(loc, adjacent);
        // Expose graph-local structure that planar convolutions cannot see across the cut seams.
        if(numAdjacent == 5)
          setRowBin(rowBin, pos, 4, 1.0f, posStride, featureStride);
        if(stone == C_EMPTY) {
          int plaNeighbors = 0;
          int oppNeighbors = 0;
          int plaSides = board.sideMask(loc);
          int oppSides = plaSides;
          for(int i = 0; i < numAdjacent; i++) {
            Loc next = adjacent[i];
            if(board.colors[next] == pla) {
              plaNeighbors++;
              plaSides |= plaComponentSides[plaComponent[next]];
            }
            else if(board.colors[next] == opp) {
              oppNeighbors++;
              oppSides |= oppComponentSides[oppComponent[next]];
            }
          }
          auto atLeastTwoSides = [](int sides) {
            return (sides & (sides - 1)) != 0;
          };
          if(plaNeighbors >= 1) setRowBin(rowBin, pos, 5, 1.0f, posStride, featureStride);
          if(plaNeighbors >= 2) setRowBin(rowBin, pos, 6, 1.0f, posStride, featureStride);
          if(oppNeighbors >= 1) setRowBin(rowBin, pos, 7, 1.0f, posStride, featureStride);
          if(oppNeighbors >= 2) setRowBin(rowBin, pos, 8, 1.0f, posStride, featureStride);
          if(atLeastTwoSides(plaSides)) setRowBin(rowBin, pos, 9, 1.0f, posStride, featureStride);
          if(atLeastTwoSides(oppSides)) setRowBin(rowBin, pos, 10, 1.0f, posStride, featureStride);
          if(plaSides == 7) setRowBin(rowBin, pos, 11, 1.0f, posStride, featureStride);
          if(oppSides == 7) setRowBin(rowBin, pos, 12, 1.0f, posStride, featureStride);
        }
      }

    }
  }

  //policyLocalFocus
  if(
    nnInputParams.policyLocalFocusPow > 0 && hist.moveHistory.size() >= 1 &&
    board.isOnBoard(hist.moveHistory[hist.moveHistory.size() - 1].loc)) {
    Loc lastMove = hist.moveHistory[hist.moveHistory.size() - 1].loc;

    int pos = NNPos::locToPos(lastMove, board.x_size, nnXLen, nnYLen);
    setRowBin(rowBin, pos, 3, 1.0f, posStride, featureStride);
    rowGlobal[1] = 1.0;
    rowGlobal[2] = nnInputParams.policyLocalFocusPow * 3.0;
    rowGlobal[3] = 3.0 / nnInputParams.policyLocalFocusDist;
  }

  rowGlobal[0] = nextPlayer == C_WHITE ? 1.0 : 0.0;

  //Global features.
  //The first 5 of them were set already above to flag which of the past 5 moves were passes.

  //Scoring
  if(hist.rules.scoringRule == Rules::SCORING_AREA) {}
  else
    ASSERT_UNREACHABLE;

  int boardArea = board.playableArea();
  if(hist.rules.maxMoves > 0 && hist.rules.maxMoves < boardArea) {
    rowGlobal[4] = 1.0;
    rowGlobal[14] =
      nextPlayer == P_BLACK ? -nnInputParams.noResultUtilityForWhite : nnInputParams.noResultUtilityForWhite;
    int mm = hist.rules.maxMoves;
    int movecount = board.numStonesOnBoard();
    int area = boardArea;
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
