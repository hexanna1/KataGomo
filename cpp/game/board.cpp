#include "../game/board.h"

/*
 * board.cpp
 * Originally from an unreleased project back in 2010, modified since.
 * Authors: brettharrison (original), David Wu (original and later modificationss).
 */

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

#include "../core/rand.h"
#include "../game/benty.h"

using namespace std;

static bool isYPoint(int x, int y, int xSize, int ySize) {
  return x >= 0 && y >= 0 && x < xSize && y < ySize && xSize == ySize && x + y < xSize;
}

static bool isObtuseYPoint(int x, int y, int xSize, int ySize) {
  if(x < 0 || y < 0 || x >= xSize || y >= ySize || xSize != ySize || xSize <= 0 || xSize % 2 != 1)
    return false;
  int n = (xSize - 1) / 2;
  int rx = xSize - 1 - x;
  int ry = ySize - 1 - y;
  bool inTopLeftCut = x < n && y < n && x + y < n;
  bool inBottomRightCut = rx < n && ry < n && rx + ry < n;
  return !inTopLeftCut && !inBottomRightCut;
}

static bool isPlayablePoint(BoardShape shape, int x, int y, int xSize, int ySize) {
  switch(shape) {
  case BoardShape::Y:
    return isYPoint(x, y, xSize, ySize);
  case BoardShape::ObtuseY:
    return isObtuseYPoint(x, y, xSize, ySize);
  case BoardShape::BentY:
    return x >= 0 && y >= 0 && x < xSize && y < ySize && xSize == ySize &&
      BentY::isSupportedTensorLen(xSize) && BentY::getTopology(xSize).playable[y * xSize + x];
  default:
    ASSERT_UNREACHABLE;
    return false;
  }
}

static bool isValidSizeForShape(BoardShape shape, int xSize, int ySize) {
  switch(shape) {
  case BoardShape::Y:
    return xSize == ySize;
  case BoardShape::ObtuseY:
    return xSize == ySize && xSize > 0 && xSize % 2 == 1;
  case BoardShape::BentY:
    return xSize == ySize && BentY::isSupportedTensorLen(xSize);
  default:
    ASSERT_UNREACHABLE;
    return false;
  }
}

static int playableArea(BoardShape shape, int xSize, int ySize) {
  switch(shape) {
  case BoardShape::Y:
    assert(xSize == ySize);
    return xSize * (xSize + 1) / 2;
  case BoardShape::ObtuseY: {
    assert(isValidSizeForShape(shape, xSize, ySize));
    int n = (xSize - 1) / 2;
    return xSize * ySize - n * (n + 1);
  }
  case BoardShape::BentY: {
    assert(isValidSizeForShape(shape, xSize, ySize));
    const vector<bool>& playable = BentY::getTopology(xSize).playable;
    return (int)count(playable.begin(), playable.end(), true);
  }
  default:
    ASSERT_UNREACHABLE;
    return 0;
  }
}

static int sideMask(BoardShape shape, int x, int y, int xSize, int ySize) {
  switch(shape) {
  case BoardShape::Y: {
    assert(isYPoint(x, y, xSize, ySize));
    int sidesTouched = 0;
    if(y == 0)
      sidesTouched |= 1;
    if(x == 0)
      sidesTouched |= 2;
    if(x + y == xSize - 1)
      sidesTouched |= 4;
    return sidesTouched;
  }
  case BoardShape::ObtuseY: {
    assert(isObtuseYPoint(x, y, xSize, ySize));
    int n = (xSize - 1) / 2;
    int sidesTouched = 0;
    if((y == 0 && x >= n) || (x == xSize - 1 && y <= n))
      sidesTouched |= 1;
    if((x + y == n) || (x == 0 && y >= n))
      sidesTouched |= 2;
    if((y == ySize - 1 && x <= n) || ((xSize - 1 - x) + (ySize - 1 - y) == n))
      sidesTouched |= 4;
    return sidesTouched;
  }
  case BoardShape::BentY:
    assert(isPlayablePoint(shape, x, y, xSize, ySize));
    return BentY::getTopology(xSize).sideMasks[y * xSize + x];
  default:
    ASSERT_UNREACHABLE;
    return 0;
  }
}

static int printableColumnsInRow(BoardShape shape, int xSize, int ySize, int y) {
  switch(shape) {
  case BoardShape::Y:
    (void)ySize;
    return xSize - y;
  case BoardShape::ObtuseY:
  case BoardShape::BentY:
    (void)ySize;
    (void)y;
    return xSize;
  default:
    ASSERT_UNREACHABLE;
    return 0;
  }
}

string BoardShapeIO::toString(BoardShape shape) {
  switch(shape) {
  case BoardShape::Y:
    return "y";
  case BoardShape::ObtuseY:
    return "obtuseY";
  case BoardShape::BentY:
    return "bentY";
  default:
    ASSERT_UNREACHABLE;
    return "";
  }
}

bool BoardShapeIO::tryParse(const string& s, BoardShape& shape) {
  string lower = Global::toLower(s);
  if(lower == "y") {
    shape = BoardShape::Y;
    return true;
  }
  if(lower == "obtusey") {
    shape = BoardShape::ObtuseY;
    return true;
  }
  if(lower == "benty") {
    shape = BoardShape::BentY;
    return true;
  }
  return false;
}

BoardShape BoardShapeIO::parse(const string& s) {
  BoardShape shape;
  if(tryParse(s, shape))
    return shape;
  throw StringError("Could not parse board shape: " + s);
}

//STATIC VARS-----------------------------------------------------------------------------
bool Board::IS_ZOBRIST_INITALIZED = false;
Hash128 Board::ZOBRIST_SIZE_X_HASH[MAX_LEN+1];
Hash128 Board::ZOBRIST_SIZE_Y_HASH[MAX_LEN+1];
Hash128 Board::ZOBRIST_BOARD_SHAPE_HASH[3];
Hash128 Board::ZOBRIST_BOARD_HASH[MAX_ARR_SIZE][4];
Hash128 Board::ZOBRIST_PLAYER_HASH[4];
Hash128 Board::ZOBRIST_MOVENUM_HASH[MAX_ARR_SIZE];
Hash128 Board::ZOBRIST_LASTMOVE_HASH[MAX_ARR_SIZE];
Hash128 Board::ZOBRIST_BOARD_HASH2[MAX_ARR_SIZE][4];
const Hash128 Board::ZOBRIST_GAME_IS_OVER = //Based on sha256 hash of Board::ZOBRIST_GAME_IS_OVER
  Hash128(0xb6f9e465597a77eeULL, 0xf1d583d960a4ce7fULL);

bool Board::IS_CAPTURETABLE_INITALIZED = false;
int8_t Board::CAPTURE_TABLE[4096];
//LOCATION--------------------------------------------------------------------------------
Loc Location::getLoc(int x, int y, int x_size)
{
  return (x+1) + (y+1)*(x_size+1);
}
int Location::getX(Loc loc, int x_size)
{
  return (loc % (x_size+1)) - 1;
}
int Location::getY(Loc loc, int x_size)
{
  return (loc / (x_size+1)) - 1;
}
void Location::getAdjacentOffsets(short adj_offsets[8], int x_size)
{
  //first 6 are connections on the triangular hex grid
  adj_offsets[0] = -(x_size+1);
  adj_offsets[1] = -1;
  adj_offsets[2] = 1;
  adj_offsets[3] = (x_size+1);
  adj_offsets[4] = -(x_size + 1) + 1;
  adj_offsets[5] = (x_size + 1) - 1;
  adj_offsets[6] = - (x_size + 1) - 1;
  adj_offsets[7] = (x_size + 1) + 1;
}

bool Location::isAdjacent(Loc loc0, Loc loc1, int x_size)
{
  return loc0 == loc1 - (x_size+1) || loc0 == loc1 - 1 || loc0 == loc1 + 1 || loc0 == loc1 + (x_size+1) ||
         loc0 == loc1 - (x_size+1) + 1 || loc0 == loc1 + (x_size+1) - 1;
}

//CONSTRUCTORS AND INITIALIZATION----------------------------------------------------------

Board::Board()
{
  init(DEFAULT_LEN,DEFAULT_LEN,BoardShape::Y);
}

Board::Board(int x, int y)
{
  init(x,y,BoardShape::Y);
}

Board::Board(int x, int y, BoardShape boardShape)
{
  init(x,y,boardShape);
}


Board::Board(const Board& other)
{
  x_size = other.x_size;
  y_size = other.y_size;
  shape = other.shape;

  memcpy(colors, other.colors, sizeof(Color)*MAX_ARR_SIZE);

  movenum = other.movenum;
  stonenum = other.stonenum;
  pos_hash = other.pos_hash;

  memcpy(adj_offsets, other.adj_offsets, sizeof(short)*8);
}

void Board::init(int xS, int yS, BoardShape boardShape)
{
  assert(IS_ZOBRIST_INITALIZED);
  if(xS < 0 || yS < 0 || xS > MAX_LEN || yS > MAX_LEN)
    throw StringError("Board::init - invalid board size");
  if(!::isValidSizeForShape(boardShape, xS, yS))
    throw StringError("Board::init - invalid board size for " + BoardShapeIO::toString(boardShape));

  x_size = xS;
  y_size = yS;
  shape = boardShape;

  for(int i = 0; i < MAX_ARR_SIZE; i++)
    colors[i] = C_WALL;

  movenum = 0;
  stonenum = 0;

  for(int y = 0; y < y_size; y++)
  {
    for(int x = 0; x < x_size; x++)
    {
      Loc loc = (x+1) + (y+1)*(x_size+1);
      if(isPlayablePoint(x,y))
        colors[loc] = C_EMPTY;
      // empty_list.add(loc);
    }
  }

  pos_hash = ZOBRIST_SIZE_X_HASH[x_size] ^ ZOBRIST_SIZE_Y_HASH[y_size] ^ ZOBRIST_BOARD_SHAPE_HASH[(int)shape];

  Location::getAdjacentOffsets(adj_offsets,x_size);
}


void Board::initHash()
{
  if(!IS_CAPTURETABLE_INITALIZED)
    initCaptureTable();
  if(IS_ZOBRIST_INITALIZED)
    return;
  Rand rand("Board::initHash()");

  auto nextHash = [&rand]() {
    uint64_t h0 = rand.nextUInt64();
    uint64_t h1 = rand.nextUInt64();
    return Hash128(h0,h1);
  };

  for(int i = 0; i<4; i++)
    ZOBRIST_PLAYER_HASH[i] = nextHash();

  //Do this second so that the player and encore hashes are not
  //afffected by the size of the board we compile with.
  for(int i = 0; i<MAX_ARR_SIZE; i++) {
    for(Color j = 0; j<4; j++) {
      if(j == C_EMPTY || j == C_WALL)
        ZOBRIST_BOARD_HASH[i][j] = Hash128();
      else
        ZOBRIST_BOARD_HASH[i][j] = nextHash();

    }
  }

  for(int i = 0; i < MAX_ARR_SIZE; i++) {
    ZOBRIST_MOVENUM_HASH[i] = nextHash();
    ZOBRIST_LASTMOVE_HASH[i] = nextHash();
  }
  ZOBRIST_MOVENUM_HASH[0] = Hash128();

  //Reseed the random number generator so that these size hashes are also
  //not affected by the size of the board we compile with
  rand.init("Board::initHash() for ZOBRIST_SIZE hashes");
  for(int i = 0; i<MAX_LEN+1; i++) {
    ZOBRIST_SIZE_X_HASH[i] = nextHash();
    ZOBRIST_SIZE_Y_HASH[i] = nextHash();
  }
  for(int i = 0; i<3; i++)
    ZOBRIST_BOARD_SHAPE_HASH[i] = nextHash();

  //Reseed and compute one more set of zobrist hashes, mixed a bit differently
  rand.init("Board::initHash() for second set of ZOBRIST hashes");
  for(int i = 0; i<MAX_ARR_SIZE; i++) {
    for(Color j = 0; j<4; j++) {
      ZOBRIST_BOARD_HASH2[i][j] = nextHash();
      ZOBRIST_BOARD_HASH2[i][j].hash0 = Hash::murmurMix(ZOBRIST_BOARD_HASH2[i][j].hash0);
      ZOBRIST_BOARD_HASH2[i][j].hash1 = Hash::splitMix64(ZOBRIST_BOARD_HASH2[i][j].hash1);
    }
  }

  IS_ZOBRIST_INITALIZED = true;
}


bool Board::isOnBoard(Loc loc) const {
  return loc >= 0 && loc < MAX_ARR_SIZE && colors[loc] != C_WALL;
}

bool Board::isPlayablePoint(int x, int y) const {
  return ::isPlayablePoint(shape, x, y, x_size, y_size);
}

int Board::playableArea() const {
  return ::playableArea(shape, x_size, y_size);
}

int Board::sideMask(Loc loc) const {
  int x = Location::getX(loc, x_size);
  int y = Location::getY(loc, x_size);
  if(!isPlayablePoint(x, y))
    return 0;
  return ::sideMask(shape, x, y, x_size, y_size);
}

int Board::getAdjacentLocs(Loc loc, Loc buf[6]) const {
  if(!isOnBoard(loc))
    return 0;
  if(shape == BoardShape::BentY) {
    int x = Location::getX(loc,x_size);
    int y = Location::getY(loc,x_size);
    int pos = y * x_size + x;
    const vector<int>& adjacent = BentY::getTopology(x_size).neighbors[pos];
    for(int i = 0; i < (int)adjacent.size(); i++) {
      int adjPos = adjacent[i];
      buf[i] = Location::getLoc(adjPos % x_size, adjPos / x_size, x_size);
    }
    return (int)adjacent.size();
  }
  for(int i = 0; i < 6; i++)
    buf[i] = loc + adj_offsets[i];
  return 6;
}

bool Board::isValidSizeForShape(int xSize, int ySize, BoardShape shape) {
  return ::isValidSizeForShape(shape, xSize, ySize);
}

//Check if moving here is illegal.
bool Board::isLegal(Loc loc, Player pla) const
{
  if(pla != P_BLACK && pla != P_WHITE)
    return false;
  return loc == PASS_LOC || (
    loc >= 0 &&
    loc < MAX_ARR_SIZE &&
    (colors[loc] == C_EMPTY)
  );
}

bool Board::isEmpty() const {
  for(int y = 0; y < y_size; y++) {
    for(int x = 0; x < x_size; x++) {
      Loc loc = Location::getLoc(x,y,x_size);
      if(colors[loc] == C_BLACK || colors[loc] == C_WHITE)
        return false;
    }
  }
  return true;
}

int Board::numStonesOnBoard() const {
  int num = 0;
  for(int y = 0; y < y_size; y++) {
    for(int x = 0; x < x_size; x++) {
      Loc loc = Location::getLoc(x,y,x_size);
      if(colors[loc] == C_BLACK || colors[loc] == C_WHITE)
        num += 1;
    }
  }
  return num;
}

int Board::numPlaStonesOnBoard(Player pla) const {
  int num = 0;
  for(int y = 0; y < y_size; y++) {
    for(int x = 0; x < x_size; x++) {
      Loc loc = Location::getLoc(x,y,x_size);
      if(colors[loc] == pla)
        num += 1;
    }
  }
  return num;
}


bool Board::setStone(Loc loc, Color color)
{
  if(loc < 0 || loc >= MAX_ARR_SIZE || colors[loc] == C_WALL)
    return false;
  if(color != C_BLACK && color != C_WHITE && color != C_EMPTY)
    return false;

  Color colorOld = colors[loc];
  colors[loc] = color;
  pos_hash ^= ZOBRIST_BOARD_HASH[loc][colorOld];
  pos_hash ^= ZOBRIST_BOARD_HASH[loc][color];

  if(colorOld != C_EMPTY)
    stonenum--;
  if(color != C_EMPTY)
    stonenum++;

  return true;
}
bool Board::setStones(std::vector<Move> placements) {
  std::set<Loc> locs;
  for(const Move& placement: placements) {
    if(locs.find(placement.loc) != locs.end())
      return false;
    locs.insert(placement.loc);
  }
  // First empty out all locations that we plan to set.
  // This guarantees avoiding any intermediate liberty issues.
  for(const Move& placement: placements) {
    bool suc = setStone(placement.loc, C_EMPTY);
    if(!suc)
      return false;
  }
  // Now set all the stones we wanted.
  for(const Move& placement: placements) {
    bool suc = setStone(placement.loc, placement.pla);
    if(!suc)
      return false;
  }
  return true;
}

//Plays the specified move, assuming it is legal.
void Board::playMoveAssumeLegal(Loc loc, Player pla)
{
  pos_hash ^= ZOBRIST_MOVENUM_HASH[movenum];
  movenum++;
  pos_hash ^= ZOBRIST_MOVENUM_HASH[movenum];

  //Pass?
  if(loc == PASS_LOC)
  {
    return;
  }
  bool suc = setStone(loc, pla);
  assert(suc);
  (void)suc;

}

Hash128 Board::getSitHash(Player pla) const {
  Hash128 h = pos_hash;
  h ^= Board::ZOBRIST_PLAYER_HASH[pla];
  return h;
}


//TACTICAL STUFF--------------------------------------------------------------------


void Board::checkConsistency() const {
  const string errLabel = string("Board::checkConsistency(): ");


  vector<Loc> buf;
  Hash128 tmp_pos_hash = ZOBRIST_SIZE_X_HASH[x_size] ^ ZOBRIST_SIZE_Y_HASH[y_size] ^ ZOBRIST_BOARD_SHAPE_HASH[(int)shape];
  for(Loc loc = 0; loc < MAX_ARR_SIZE; loc++) {
    int x = Location::getX(loc,x_size);
    int y = Location::getY(loc,x_size);
    if(x < 0 || x >= x_size || y < 0 || y >= y_size || !isPlayablePoint(x,y)) {
      if(colors[loc] != C_WALL)
        throw StringError(errLabel + "Non-WALL value outside of board legal area");
    }
    else {
      if(colors[loc] == C_BLACK || colors[loc] == C_WHITE) {
        tmp_pos_hash ^= ZOBRIST_BOARD_HASH[loc][colors[loc]];
        tmp_pos_hash ^= ZOBRIST_BOARD_HASH[loc][C_EMPTY];
      }
      else if(colors[loc] != C_EMPTY)
        throw StringError(errLabel + "Non-(black,white,empty) value within board legal area");
    }
  }

  tmp_pos_hash ^= ZOBRIST_MOVENUM_HASH[movenum];

  if(pos_hash != tmp_pos_hash)
    throw StringError(errLabel + "Pos hash does not match expected");

  if(stonenum != numStonesOnBoard())
    throw StringError(errLabel + "stoneNum does not match expected");


  short tmpAdjOffsets[8];
  Location::getAdjacentOffsets(tmpAdjOffsets,x_size);
  for(int i = 0; i<8; i++)
    if(tmpAdjOffsets[i] != adj_offsets[i])
      throw StringError(errLabel + "Corrupted adj_offsets array");
}

bool Board::isEqualForTesting(const Board& other) const {
  checkConsistency();
  other.checkConsistency();
  if(x_size != other.x_size)
    return false;
  if(y_size != other.y_size)
    return false;
  if(shape != other.shape)
    return false;
  if(pos_hash != other.pos_hash)
    return false;
  for(int i = 0; i<MAX_ARR_SIZE; i++) {
    if(colors[i] != other.colors[i])
      return false;
  }
  //We don't require that the chain linked lists are in the same order.
  //Consistency check ensures that all the linked lists are consistent with colors array, which we checked.
  return true;
}



//IO FUNCS------------------------------------------------------------------------------------------

char PlayerIO::colorToChar(Color c)
{
  switch(c) {
  case C_BLACK: return 'X';
  case C_WHITE: return 'O';
  case C_EMPTY: return '.';
  default:  return '#';
  }
}

string PlayerIO::playerToString(Color c)
{
  switch(c) {
  case C_BLACK: return "Black";
  case C_WHITE: return "White";
  case C_EMPTY: return "Empty";
  default:  return "Wall";
  }
}

string PlayerIO::playerToStringShort(Color c)
{
  switch(c) {
  case C_BLACK: return "B";
  case C_WHITE: return "W";
  case C_EMPTY: return "E";
  default:  return "";
  }
}

bool PlayerIO::tryParsePlayer(const string& s, Player& pla) {
  string str = Global::toLower(s);
  if(str == "black" || str == "b") {
    pla = P_BLACK;
    return true;
  }
  else if(str == "white" || str == "w") {
    pla = P_WHITE;
    return true;
  }
  return false;
}

Player PlayerIO::parsePlayer(const string& s) {
  Player pla = C_EMPTY;
  bool suc = tryParsePlayer(s,pla);
  if(!suc)
    throw StringError("Could not parse player: " + s);
  return pla;
}

string Location::toStringMach(Loc loc, int x_size)
{
  if(loc == Board::PASS_LOC)
    return string("pass");
  if(loc == Board::NULL_LOC)
    return string("null");

  int x = getX(loc, x_size), y = getY(loc, x_size);

  char buf[128];
  snprintf(buf, sizeof(buf), "(%d,%d)", x, y);
  return string(buf);
}

static string humanLettersForX(int x) {
  string s;
  int col = x + 1;
  while(col > 0) {
    col -= 1;
    s += (char)('a' + (col % 26));
    col /= 26;
  }
  std::reverse(s.begin(), s.end());
  return s;
}

static bool isHumanCoordLetter(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

static bool tryParseHumanLetters(const string& s, size_t& pos, int& x) {
  size_t start = pos;
  int col = 0;
  while(pos < s.length() && isHumanCoordLetter(s[pos])) {
    char c = s[pos];
    if(c >= 'A' && c <= 'Z')
      c = (char)(c - 'A' + 'a');
    col = col * 26 + (c - 'a' + 1);
    pos++;
  }
  if(pos == start)
    return false;
  x = col - 1;
  return true;
}

static bool rowLooksSpaced(const string& line, int columns) {
  if(columns <= 0 || line.length() != (size_t)(2 * columns - 1))
    return false;
  for(size_t i = 1; i < line.length(); i += 2)
    if(line[i] != ' ')
      return false;
  return true;
}

static string locationToString(Loc loc, int x_size, int y_size, BoardShape shape)
{
  if(loc == Board::PASS_LOC)
    return string("pass");
  if(loc == Board::NULL_LOC)
    return string("null");
  int x = Location::getX(loc,x_size);
  int y = Location::getY(loc,x_size);
  if(!isPlayablePoint(shape,x,y,x_size,y_size))
    return Location::toStringMach(loc,x_size);
  return humanLettersForX(x) + std::to_string(y + 1);
}

string Location::toString(Loc loc, int x_size, int y_size)
{
  return locationToString(loc, x_size, y_size, BoardShape::Y);
}

string Location::toString(Loc loc, const Board& b) {
  return locationToString(loc,b.x_size,b.y_size,b.shape);
}

string Location::toStringMach(Loc loc, const Board& b) {
  return toStringMach(loc,b.x_size);
}

static bool tryLocationOfString(const string& str, int x_size, int y_size, BoardShape shape, Loc& result) {
  string s = Global::trim(str);
  if(s.length() < 2)
    return false;
  if(Global::isEqualCaseInsensitive(s,string("pass")) || Global::isEqualCaseInsensitive(s,string("pss"))) {
    result = Board::PASS_LOC;
    return true;
  }
  if(s[0] == '(') {
    if(s[s.length()-1] != ')')
      return false;
    s = s.substr(1,s.length()-2);
    vector<string> pieces = Global::split(s,',');
    if(pieces.size() != 2)
      return false;
    int x;
    int y;
    bool sucX = Global::tryStringToInt(pieces[0],x);
    bool sucY = Global::tryStringToInt(pieces[1],y);
    if(!sucX || !sucY)
      return false;
    if(!isPlayablePoint(shape,x,y,x_size,y_size))
      return false;
    result = Location::getLoc(x,y,x_size);
    return true;
  }
  else {
    int x;
    size_t pos = 0;
    if(!tryParseHumanLetters(s,pos,x))
      return false;
    s = s.substr(pos,s.length()-pos);

    int y;
    bool sucY = Global::tryStringToInt(s,y);
    if(!sucY)
      return false;
    y -= 1;
    if(!isPlayablePoint(shape,x,y,x_size,y_size))
      return false;
    result = Location::getLoc(x,y,x_size);
    return true;
  }
}

bool Location::tryOfString(const string& str, int x_size, int y_size, Loc& result) {
  return tryLocationOfString(str, x_size, y_size, BoardShape::Y, result);
}

bool Location::tryOfStringAllowNull(const string& str, int x_size, int y_size, Loc& result) {
  if(str == "null") {
    result = Board::NULL_LOC;
    return true;
  }
  return tryOfString(str, x_size, y_size, result);
}

bool Location::tryOfString(const string& str, const Board& b, Loc& result) {
  return tryLocationOfString(str,b.x_size,b.y_size,b.shape,result);
}

bool Location::tryOfStringAllowNull(const string& str, const Board& b, Loc& result) {
  if(str == "null") {
    result = Board::NULL_LOC;
    return true;
  }
  return tryOfString(str,b,result);
}

Loc Location::ofString(const string& str, int x_size, int y_size) {
  Loc result;
  if(tryOfString(str,x_size,y_size,result))
    return result;
  throw StringError("Could not parse board location: " + str);
}

Loc Location::ofStringAllowNull(const string& str, int x_size, int y_size) {
  Loc result;
  if(tryOfStringAllowNull(str,x_size,y_size,result))
    return result;
  throw StringError("Could not parse board location: " + str);
}

Loc Location::ofString(const string& str, const Board& b) {
  Loc result;
  if(tryOfString(str,b,result))
    return result;
  throw StringError("Could not parse board location: " + str);
}


Loc Location::ofStringAllowNull(const string& str, const Board& b) {
  Loc result;
  if(tryOfStringAllowNull(str,b,result))
    return result;
  throw StringError("Could not parse board location: " + str);
}

vector<Loc> Location::parseSequence(const string& str, const Board& board) {
  vector<string> pieces = Global::split(Global::trim(str),' ');
  vector<Loc> locs;
  for(size_t i = 0; i<pieces.size(); i++) {
    string piece = Global::trim(pieces[i]);
    if(piece.length() <= 0)
      continue;
    locs.push_back(Location::ofString(piece,board));
  }
  return locs;
}

void Board::printBoard(ostream& out, const Board& board, Loc markLoc, const vector<Move>* hist) {
  if(hist != NULL)
    out << "MoveNum: " << hist->size() << " ";
  out << "HASH: " << board.pos_hash << "\n";
  bool showCoords = board.x_size <= 50 && board.y_size <= 50;
  if(showCoords) {
    if(humanLettersForX(board.x_size-1).length() > 1) {
      out << "   ";
      for(int x = 0; x < board.x_size; x++) {
        string column = humanLettersForX(x);
        out << (column.length() > 1 ? column[0] : ' ');
        if(x < board.x_size-1)
          out << ' ';
      }
      out << "\n";
    }
    out << "   ";
    for(int x = 0; x < board.x_size; x++) {
      string column = humanLettersForX(x);
      out << column[column.length()-1];
      if(x < board.x_size-1)
        out << " ";
    }
    out << "\n";
  }

  for(int y = 0; y < board.y_size; y++)
  {
    if(showCoords) {
      char buf[16];
      snprintf(buf, sizeof(buf), "%2d", y+1);
      out << buf << ' ';
    }
    for(int i = 0; i < y; i++)
      out << ' ';

    int playableWidth = printableColumnsInRow(board.shape, board.x_size, board.y_size, y);
    for(int x = 0; x < playableWidth; x++)
    {
      Loc loc = Location::getLoc(x,y,board.x_size);
      char s = PlayerIO::colorToChar(board.colors[loc]);
      if(board.colors[loc] == C_EMPTY && markLoc == loc)
        out << '@';
      else
        out << s;

      bool histMarked = false;
      if(hist != NULL) {
        size_t start = hist->size() >= 3 ? hist->size()-3 : 0;
        for(size_t i = 0; start+i < hist->size(); i++) {
          if((*hist)[start+i].loc == loc) {
            out << (1+i);
            histMarked = true;
            break;
          }
        }
      }

      if(x < playableWidth-1 && !histMarked)
        out << ' ';
    }
    out << "\n";
  }
  out << "\n";
}

ostream& operator<<(ostream& out, const Board& board) {
  Board::printBoard(out,board,Board::NULL_LOC,NULL);
  return out;
}


string Board::toStringSimple(const Board& board, char lineDelimiter) {
  string s;
  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      s += PlayerIO::colorToChar(board.colors[loc]);
    }
    s += lineDelimiter;
  }
  return s;
}

Board Board::parseBoard(int xSize, int ySize, const string& s) {
  return parseBoard(xSize,ySize,s,'\n');
}

Board Board::parseBoard(int xSize, int ySize, BoardShape shape, const string& s) {
  return parseBoard(xSize,ySize,shape,s,'\n');
}

Board Board::parseBoard(int xSize, int ySize, const string& s, char lineDelimiter) {
  return parseBoard(xSize,ySize,BoardShape::Y,s,lineDelimiter);
}

Board Board::parseBoard(int xSize, int ySize, BoardShape shape, const string& s, char lineDelimiter) {
  Board board(xSize,ySize,shape);
  vector<string> lines = Global::split(Global::trim(s),lineDelimiter);

  //Throw away coordinate labels line if it exists
  if(lines.size() == ySize+1 && Global::isPrefix(Global::toLower(Global::trim(lines[0])),"a"))
    lines.erase(lines.begin());

  if(lines.size() != ySize)
    throw StringError("Board::parseBoard - string has different number of board rows than ySize");

  for(int y = 0; y<ySize; y++) {
    string line = Global::trim(lines[y]);
    //Throw away coordinates if they exist
    size_t firstNonDigitIdx = 0;
    while(firstNonDigitIdx < line.length() && Global::isDigit(line[firstNonDigitIdx]))
      firstNonDigitIdx++;
    line.erase(0,firstNonDigitIdx);
    line = Global::trim(line);

    int playableWidth = printableColumnsInRow(shape, xSize, ySize, y);
    bool rectangularNoSpaces = line.length() == (size_t)xSize;
    bool rectangularSpaced = rowLooksSpaced(line,xSize);
    bool triangularNoSpaces = line.length() == (size_t)playableWidth;
    bool triangularSpaced = rowLooksSpaced(line,playableWidth);
    if(!rectangularNoSpaces && !rectangularSpaced && !triangularNoSpaces && !triangularSpaced)
      throw StringError("Board::parseBoard - line length not compatible with xSize");

    bool spaced = false;
    int columnsToRead = xSize;
    if(triangularSpaced && (!rectangularSpaced || playableWidth != xSize)) {
      spaced = true;
      columnsToRead = playableWidth;
    }
    else if(rectangularSpaced) {
      spaced = true;
      columnsToRead = xSize;
    }
    else if(triangularNoSpaces && (!rectangularNoSpaces || playableWidth != xSize)) {
      spaced = false;
      columnsToRead = playableWidth;
    }
    else if(rectangularNoSpaces) {
      spaced = false;
      columnsToRead = xSize;
    }
    else if(triangularSpaced) {
      spaced = true;
      columnsToRead = playableWidth;
    }
    else if(triangularNoSpaces) {
      spaced = false;
      columnsToRead = playableWidth;
    }
    for(int x = 0; x<columnsToRead; x++) {
      char c;
      if(!spaced)
        c = line[x];
      else
        c = line[x*2];

      Loc loc = Location::getLoc(x,y,board.x_size);
      if(!board.isPlayablePoint(x,y)) {
        if(c != '#')
          throw StringError(string("Board::parseBoard - non-wall character outside playable area near ") + Location::toStringMach(loc,board));
        continue;
      }
      if(c == '#') {
        throw StringError(string("Board::parseBoard - wall on playable point near ") + Location::toString(loc,board));
      }
      if(c == '.' || c == ' ' || c == '*' || c == ',' || c == '`')
        continue;
      else if(c == 'o' || c == 'O') {
        bool suc = board.setStone(loc,P_WHITE);
        if(!suc)
          throw StringError(string("Board::parseBoard - zero-liberty group near ") + Location::toString(loc,board));
      }
      else if(c == 'x' || c == 'X') {
        bool suc = board.setStone(loc,P_BLACK);
        if(!suc)
          throw StringError(string("Board::parseBoard - zero-liberty group near ") + Location::toString(loc,board));
      }
      else
        throw StringError(string("Board::parseBoard - could not parse board character: ") + c);
    }
  }
  return board;
}

nlohmann::json Board::toJson(const Board& board) {
  nlohmann::json data;
  data["xSize"] = board.x_size;
  data["ySize"] = board.y_size;
  data["shape"] = BoardShapeIO::toString(board.shape);
  data["stones"] = Board::toStringSimple(board,'|');
  return data;
}

Board Board::ofJson(const nlohmann::json& data) {
  int xSize = data["xSize"].get<int>();
  int ySize = data["ySize"].get<int>();
  BoardShape shape = BoardShapeIO::parse(data["shape"].get<string>());
  Board board = Board::parseBoard(xSize,ySize,shape,data["stones"].get<string>(),'|');
  return board;
}
