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

using namespace std;

//STATIC VARS-----------------------------------------------------------------------------
bool Board::IS_ZOBRIST_INITALIZED = false;
Hash128 Board::ZOBRIST_SIZE_X_HASH[MAX_LEN+1];
Hash128 Board::ZOBRIST_SIZE_Y_HASH[MAX_LEN+1];
Hash128 Board::ZOBRIST_BOARD_HASH[MAX_ARR_SIZE][4];
Hash128 Board::ZOBRIST_PLAYER_HASH[4];
Hash128 Board::ZOBRIST_MOVENUM_HASH[MAX_ARR_SIZE];
Hash128 Board::ZOBRIST_LASTMOVE_HASH[MAX_ARR_SIZE];
Hash128 Board::ZOBRIST_BOARD_HASH2[MAX_ARR_SIZE][4];
Hash128 Board::ZOBRIST_CROSSCUT_DIRECTION_HASH[MAX_ARR_SIZE][4];
Hash128 Board::ZOBRIST_VARIANT_HASH[3];
const Hash128 Board::ZOBRIST_GAME_IS_OVER = //Based on sha256 hash of Board::ZOBRIST_GAME_IS_OVER
  Hash128(0xb6f9e465597a77eeULL, 0xf1d583d960a4ce7fULL);

bool Board::IS_CAPTURETABLE_INITALIZED = false;
int8_t Board::CAPTURE_TABLE[4096];

string QuaxVariantIO::toString(QuaxVariant variant) {
  switch(variant) {
  case QuaxVariant::DoubleCrosscut: return "double";
  case QuaxVariant::SingleCrosscut: return "single";
  case QuaxVariant::Official: return "official";
  default: ASSERT_UNREACHABLE;
  }
}

bool QuaxVariantIO::tryParse(const string& s, QuaxVariant& variant) {
  string lower = Global::toLower(Global::trim(s));
  if(lower == "double") {
    variant = QuaxVariant::DoubleCrosscut;
    return true;
  }
  if(lower == "single") {
    variant = QuaxVariant::SingleCrosscut;
    return true;
  }
  if(lower == "official") {
    variant = QuaxVariant::Official;
    return true;
  }
  return false;
}

QuaxVariant QuaxVariantIO::parse(const string& s) {
  QuaxVariant variant;
  if(!tryParse(s,variant))
    throw StringError("Unknown Quax variant: " + s);
  return variant;
}

bool QuaxVariantIO::hasDirectionalCrosscuts(QuaxVariant variant) {
  return variant == QuaxVariant::SingleCrosscut || variant == QuaxVariant::Official;
}
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
  //first 6 are connections on Hex board
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
  return loc0 == loc1 - (x_size+1) || loc0 == loc1 - 1 || loc0 == loc1 + 1 || loc0 == loc1 + (x_size+1);
}



#define FOREACHADJ(BLOCK) {int ADJOFFSET = -(x_size+1); {BLOCK}; ADJOFFSET = -1; {BLOCK}; ADJOFFSET = 1; {BLOCK}; ADJOFFSET = x_size+1; {BLOCK}};
#define ADJ0 (-(x_size+1))
#define ADJ1 (-1)
#define ADJ2 (1)
#define ADJ3 (x_size+1)

//CONSTRUCTORS AND INITIALIZATION----------------------------------------------------------

Board::Board()
{
  init(DEFAULT_LEN,internalYSizeForUserSize(DEFAULT_LEN),QuaxVariant::DoubleCrosscut);
}

Board::Board(int x, int y)
{
  init(x,y,QuaxVariant::DoubleCrosscut);
}

Board::Board(int x, int y, QuaxVariant v)
{
  init(x,y,v);
}


Board::Board(const Board& other)
{
  x_size = other.x_size;
  y_size = other.y_size;
  variant = other.variant;

  memcpy(colors, other.colors, sizeof(Color)*MAX_ARR_SIZE);
  memcpy(crosscutDirections, other.crosscutDirections, sizeof(CrosscutDirection)*MAX_ARR_SIZE);

  movenum = other.movenum;
  stonenum = other.stonenum;
  pos_hash = other.pos_hash;

  memcpy(adj_offsets, other.adj_offsets, sizeof(short)*8);
}

void Board::init(int xS, int yS, QuaxVariant v)
{
  assert(IS_ZOBRIST_INITALIZED);
  if(xS < 0 || yS < 0 || xS > MAX_LEN || yS > MAX_LEN || !isValidQuaxDimensions(xS,yS))
    throw StringError("Board::init - invalid Quax board dimensions");

  x_size = xS;
  y_size = yS;
  variant = v;

  for(int i = 0; i < MAX_ARR_SIZE; i++) {
    colors[i] = C_WALL;
    crosscutDirections[i] = CROSSCUT_NONE;
  }

  movenum = 0;
  stonenum = 0;

  for(int y = 0; y < y_size; y++)
  {
    for(int x = 0; x < x_size; x++)
    {
      Loc loc = (x+1) + (y+1)*(x_size+1);
      if(y % 2 == 0 || x < x_size-1)
        colors[loc] = C_EMPTY;
      // empty_list.add(loc);
    }
  }

  pos_hash = ZOBRIST_SIZE_X_HASH[x_size] ^ ZOBRIST_SIZE_Y_HASH[y_size] ^ ZOBRIST_VARIANT_HASH[(int)variant];

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

  for(int i = 0; i<3; i++)
    ZOBRIST_VARIANT_HASH[i] = nextHash();

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

  for(int i = 0; i<MAX_ARR_SIZE; i++) {
    ZOBRIST_CROSSCUT_DIRECTION_HASH[i][CROSSCUT_NONE] = Hash128();
    for(int direction = CROSSCUT_BOTH; direction <= CROSSCUT_SLASH; direction++)
      ZOBRIST_CROSSCUT_DIRECTION_HASH[i][direction] = nextHash();
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

bool Board::isDiamond(Loc loc) const {
  return isOnBoard(loc) && Location::getY(loc,x_size) % 2 == 1;
}

bool Board::isSecondCrosscutLoc(Loc loc) const {
  if(!QuaxVariantIO::hasDirectionalCrosscuts(variant) || loc < SECOND_CROSSCUT_LOC_BASE || loc >= MAX_ARR_SIZE)
    return false;
  int index = loc-SECOND_CROSSCUT_LOC_BASE;
  return index < (x_size-1)*(x_size-1);
}

Loc Board::getPhysicalLoc(Loc loc) const {
  if(!isSecondCrosscutLoc(loc))
    return loc;
  int index = loc-SECOND_CROSSCUT_LOC_BASE;
  int x = index % (x_size-1);
  int row = index / (x_size-1);
  return Location::getLoc(x,2*row+1,x_size);
}

Loc Board::getSecondCrosscutLoc(Loc diamondLoc) const {
  assert(QuaxVariantIO::hasDirectionalCrosscuts(variant) && isDiamond(diamondLoc));
  int x = Location::getX(diamondLoc,x_size);
  int row = Location::getY(diamondLoc,x_size)/2;
  return SECOND_CROSSCUT_LOC_BASE + row*(x_size-1)+x;
}

CrosscutDirection Board::getCrosscutDirectionForMove(Loc loc) const {
  if(isSecondCrosscutLoc(loc))
    return CROSSCUT_SLASH;
  if(isDiamond(loc))
    return variant == QuaxVariant::DoubleCrosscut ? CROSSCUT_BOTH : CROSSCUT_BACKSLASH;
  return CROSSCUT_NONE;
}

int Board::playableArea() const {
  return x_size*x_size + (x_size-1)*(x_size-1);
}

bool Board::isValidQuaxDimensions(int xSize, int ySize) {
  return xSize >= 2 && ySize == internalYSizeForUserSize(xSize) && ySize <= MAX_LEN;
}

int Board::internalYSizeForUserSize(int size) {
  return 2 * size - 1;
}

//Check if moving here is illegal.
bool Board::isLegal(Loc loc, Player pla) const
{
  if(pla != P_BLACK && pla != P_WHITE)
    return false;
  bool secondCrosscut = isSecondCrosscutLoc(loc);
  Loc physicalLoc = secondCrosscut ? getPhysicalLoc(loc) : loc;
  if(variant == QuaxVariant::Official && isDiamond(physicalLoc)) {
    if(colors[physicalLoc] != C_EMPTY)
      return false;
    int x = Location::getX(physicalLoc,x_size);
    int row = Location::getY(physicalLoc,x_size)/2;
    Loc first = Location::getLoc(secondCrosscut ? x+1 : x,2*row,x_size);
    Loc second = Location::getLoc(secondCrosscut ? x : x+1,2*(row+1),x_size);
    return colors[first] == pla && colors[second] == pla;
  }
  if(secondCrosscut)
    return colors[physicalLoc] == C_EMPTY;
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
      if(isOnBoard(loc) && colors[loc] != C_EMPTY)
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
  bool secondCrosscut = isSecondCrosscutLoc(loc);
  Loc physicalLoc = secondCrosscut ? getPhysicalLoc(loc) : loc;
  if(physicalLoc < 0 || physicalLoc >= MAX_ARR_SIZE || colors[physicalLoc] == C_WALL)
    return false;
  if(secondCrosscut && !isDiamond(physicalLoc))
    return false;
  if(color != C_BLACK && color != C_WHITE && color != C_EMPTY)
    return false;

  Color colorOld = colors[physicalLoc];
  CrosscutDirection directionOld = crosscutDirections[physicalLoc];
  CrosscutDirection direction = CROSSCUT_NONE;
  if(color != C_EMPTY && isDiamond(physicalLoc))
    direction = secondCrosscut ? CROSSCUT_SLASH : variant == QuaxVariant::DoubleCrosscut ? CROSSCUT_BOTH : CROSSCUT_BACKSLASH;
  colors[physicalLoc] = color;
  crosscutDirections[physicalLoc] = direction;
  pos_hash ^= ZOBRIST_BOARD_HASH[physicalLoc][colorOld];
  pos_hash ^= ZOBRIST_BOARD_HASH[physicalLoc][color];
  pos_hash ^= ZOBRIST_CROSSCUT_DIRECTION_HASH[physicalLoc][directionOld];
  pos_hash ^= ZOBRIST_CROSSCUT_DIRECTION_HASH[physicalLoc][direction];

  if(colorOld != C_EMPTY)
    stonenum--;
  if(color != C_EMPTY)
    stonenum++;

  return true;
}
bool Board::setStones(std::vector<Move> placements) {
  std::set<Loc> locs;
  for(const Move& placement: placements) {
    Loc physicalLoc = getPhysicalLoc(placement.loc);
    if(locs.find(physicalLoc) != locs.end())
      return false;
    locs.insert(physicalLoc);
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
  setStone(loc, pla);

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
  Hash128 tmp_pos_hash = ZOBRIST_SIZE_X_HASH[x_size] ^ ZOBRIST_SIZE_Y_HASH[y_size] ^ ZOBRIST_VARIANT_HASH[(int)variant];
  for(Loc loc = 0; loc < MAX_ARR_SIZE; loc++) {
    int x = Location::getX(loc,x_size);
    int y = Location::getY(loc,x_size);
    bool playable = x >= 0 && x < x_size && y >= 0 && y < y_size && (y % 2 == 0 || x < x_size-1);
    if(!playable) {
      if(colors[loc] != C_WALL)
        throw StringError(errLabel + "Non-WALL value outside of board legal area");
    }
    else {
      if(colors[loc] == C_BLACK || colors[loc] == C_WHITE) {
        tmp_pos_hash ^= ZOBRIST_BOARD_HASH[loc][colors[loc]];
        tmp_pos_hash ^= ZOBRIST_BOARD_HASH[loc][C_EMPTY];
        CrosscutDirection expectedDirection = CROSSCUT_NONE;
        if(y % 2 == 1) {
          expectedDirection = crosscutDirections[loc];
          if(variant == QuaxVariant::DoubleCrosscut && expectedDirection != CROSSCUT_BOTH)
            throw StringError(errLabel + "Invalid double-crosscut direction");
          if(QuaxVariantIO::hasDirectionalCrosscuts(variant) && expectedDirection != CROSSCUT_BACKSLASH && expectedDirection != CROSSCUT_SLASH)
            throw StringError(errLabel + "Invalid single-crosscut direction");
        }
        else if(crosscutDirections[loc] != CROSSCUT_NONE)
          throw StringError(errLabel + "Crosscut direction on octagon");
        tmp_pos_hash ^= ZOBRIST_CROSSCUT_DIRECTION_HASH[loc][expectedDirection];
      }
      else if(colors[loc] == C_EMPTY) {
        if(crosscutDirections[loc] != CROSSCUT_NONE)
          throw StringError(errLabel + "Crosscut direction on empty location");
      }
      else
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
  if(variant != other.variant)
    return false;
  if(pos_hash != other.pos_hash)
    return false;
  for(int i = 0; i<MAX_ARR_SIZE; i++) {
    if(colors[i] != other.colors[i])
      return false;
    if(crosscutDirections[i] != other.crosscutDirections[i])
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

  int x_print = getX(loc, x_size);
  int y_print = getY(loc, x_size);

  char buf[128];
  sprintf(buf, "(%d,%d)", x_print, y_print);
  return string(buf);
}

string Location::toString(Loc loc, int x_size, int y_size)
{
  if(x_size > 26 * 5 || y_size > 26 * 5)
    return toStringMach(loc,x_size);
  if(loc == Board::PASS_LOC)
    return string("pass");
  if(loc == Board::NULL_LOC)
    return string("null");
  const char* xChar = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  int x = getX(loc,x_size);
  int y = getY(loc,x_size);
  if(x >= x_size || x < 0 || y < 0 || y >= y_size)
    return toStringMach(loc,x_size);
  char buf[128];
  int row = y / 2 + 1;
  if(x <= 25)
    sprintf(buf, y % 2 == 0 ? "%c%d" : "%c%d*", xChar[x], row);
  else
    sprintf(buf, y % 2 == 0 ? "%c%c%d" : "%c%c%d*", xChar[x / 26 - 1], xChar[x % 26], row);
  return string(buf);
}

string Location::toString(Loc loc, const Board& b) {
  if(b.isSecondCrosscutLoc(loc))
    return toString(b.getPhysicalLoc(loc),b.x_size,b.y_size) + "/";
  string s = toString(loc,b.x_size,b.y_size);
  if(QuaxVariantIO::hasDirectionalCrosscuts(b.variant) && b.isDiamond(loc))
    s += "\\";
  return s;
}

string Location::toStringMach(Loc loc, const Board& b) {
  if(b.isSecondCrosscutLoc(loc))
    return toStringMach(b.getPhysicalLoc(loc),b.x_size) + "/";
  string s = toStringMach(loc,b.x_size);
  if(QuaxVariantIO::hasDirectionalCrosscuts(b.variant) && b.isDiamond(loc))
    s += "\\";
  return s;
}

static bool tryParseLetterCoordinate(char c, int& x) {
  if(c >= 'A' && c <= 'Z')
    x = c-'A';
  else if(c >= 'a' && c <= 'z')
    x = c-'a';
  else
    return false;
  return true;
}

bool Location::tryOfString(const string& str, int x_size, int y_size, Loc& result) {
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
    if(x < 0 || y < 0 || x >= x_size || y >= y_size || (y % 2 == 1 && x == x_size-1))
      return false;
    result = Location::getLoc(x,y,x_size);
    return true;
  }
  else {
    int x;
    if(!tryParseLetterCoordinate(s[0],x))
      return false;

    //Extended format
    if((s[1] >= 'A' && s[1] <= 'Z') || (s[1] >= 'a' && s[1] <= 'z')) {
      int x1;
      if(!tryParseLetterCoordinate(s[1],x1))
        return false;
      x = (x+1) * 26 + x1;
      s = s.substr(2,s.length()-2);
    }
    else {
      s = s.substr(1,s.length()-1);
    }

    bool diamond = !s.empty() && s[s.length()-1] == '*';
    if(diamond)
      s.erase(s.length()-1);
    int y;
    bool sucY = Global::tryStringToInt(s,y);
    if(!sucY)
      return false;
    if(y < 1)
      return false;
    y = 2 * (y - 1) + (diamond ? 1 : 0);
    if(x < 0 || y < 0 || x >= x_size || y >= y_size || (diamond && x >= x_size-1))
      return false;
    result = Location::getLoc(x,y,x_size);
    return true;
  }
}

bool Location::tryOfStringAllowNull(const string& str, int x_size, int y_size, Loc& result) {
  if(str == "null") {
    result = Board::NULL_LOC;
    return true;
  }
  return tryOfString(str, x_size, y_size, result);
}

bool Location::tryOfString(const string& str, const Board& b, Loc& result) {
  string s = Global::trim(str);
  bool backslash = !s.empty() && s[s.length()-1] == '\\';
  bool slash = !s.empty() && s[s.length()-1] == '/';
  if(backslash || slash)
    s.erase(s.length()-1);
  Loc physicalLoc;
  if(!tryOfString(s,b.x_size,b.y_size,physicalLoc))
    return false;
  if(physicalLoc == Board::PASS_LOC) {
    if(backslash || slash)
      return false;
    result = physicalLoc;
    return true;
  }
  if(b.variant == QuaxVariant::DoubleCrosscut) {
    if(backslash || slash)
      return false;
    result = physicalLoc;
    return true;
  }
  if(b.isDiamond(physicalLoc)) {
    if(!backslash && !slash)
      return false;
    result = slash ? b.getSecondCrosscutLoc(physicalLoc) : physicalLoc;
    return true;
  }
  if(backslash || slash)
    return false;
  result = physicalLoc;
  return true;
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
    const char* xChar = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    out << "  ";
    for(int x = 0; x < board.x_size; x++) {
      if(x <= 25) {
        out << " ";
        out << xChar[x];
      }
      else {
        out << "A" << xChar[x-26];
      }
    }
    out << "\n";
  }

  for(int y = 0; y < board.y_size; y++)
  {
    if(showCoords) {
      char buf[16];
      sprintf(buf,"%2d",y / 2 + 1);
      out << buf << ' ';
    }
    if(y % 2 == 1)
      out << " ";
    for(int x = 0; x < board.x_size; x++)
    {
      Loc loc = Location::getLoc(x,y,board.x_size);
      char s = PlayerIO::colorToChar(board.colors[loc]);
      if(board.colors[loc] == C_EMPTY && board.getPhysicalLoc(markLoc) == loc)
        out << '@';
      else
        out << s;

      bool histMarked = false;
      if(hist != NULL) {
        size_t start = hist->size() >= 3 ? hist->size()-3 : 0;
        for(size_t i = 0; start+i < hist->size(); i++) {
          if(board.getPhysicalLoc((*hist)[start+i].loc) == loc) {
            out << (1+i);
            histMarked = true;
            break;
          }
        }
      }

      if(x < board.x_size-1 && !histMarked)
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
  return parseBoard(xSize,ySize,QuaxVariant::DoubleCrosscut,s,'\n');
}

Board Board::parseBoard(int xSize, int ySize, const string& s, char lineDelimiter) {
  return parseBoard(xSize,ySize,QuaxVariant::DoubleCrosscut,s,lineDelimiter);
}

Board Board::parseBoard(int xSize, int ySize, QuaxVariant variant, const string& s) {
  return parseBoard(xSize,ySize,variant,s,'\n');
}

Board Board::parseBoard(int xSize, int ySize, QuaxVariant variant, const string& s, char lineDelimiter) {
  Board board(xSize,ySize,variant);
  vector<string> lines = Global::split(Global::trim(s),lineDelimiter);

  //Throw away coordinate labels line if it exists
  if(lines.size() == ySize+1 && Global::isPrefix(lines[0],"A"))
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

    if(line.length() != xSize && line.length() != 2*xSize-1)
      throw StringError("Board::parseBoard - line length not compatible with xSize");

    for(int x = 0; x<xSize; x++) {
      char c;
      if(line.length() == xSize)
        c = line[x];
      else
        c = line[x*2];

      Loc loc = Location::getLoc(x,y,board.x_size);
      if(!board.isOnBoard(loc)) {
        if(c != '#')
          throw StringError("Board::parseBoard - expected wall at non-playable Quax location");
      }
      else if(c == '.' || c == ' ' || c == '*' || c == ',' || c == '`')
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
  data["variant"] = QuaxVariantIO::toString(board.variant);
  data["stones"] = Board::toStringSimple(board,'|');
  string directions;
  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      CrosscutDirection direction = board.isDiamond(loc) ? board.crosscutDirections[loc] : CROSSCUT_NONE;
      directions += direction == CROSSCUT_BOTH ? 'b' : direction == CROSSCUT_BACKSLASH ? '\\' : direction == CROSSCUT_SLASH ? '/' : '.';
    }
    directions += '|';
  }
  data["crosscuts"] = directions;
  return data;
}

Board Board::ofJson(const nlohmann::json& data) {
  int xSize = data["xSize"].get<int>();
  int ySize = data["ySize"].get<int>();
  QuaxVariant variant = QuaxVariantIO::parse(data["variant"].get<string>());
  Board board = Board::parseBoard(xSize,ySize,variant,data["stones"].get<string>(),'|');
  vector<string> directionRows = Global::split(data["crosscuts"].get<string>(),'|');
  if(directionRows.size() == (size_t)ySize+1 && directionRows.back().empty())
    directionRows.pop_back();
  if(directionRows.size() != (size_t)ySize)
    throw StringError("Board::ofJson - invalid crosscut rows");
  for(int y = 1; y < ySize; y += 2) {
    if(directionRows[y].length() != (size_t)xSize)
      throw StringError("Board::ofJson - invalid crosscut row length");
    for(int x = 0; x < xSize-1; x++) {
      Loc loc = Location::getLoc(x,y,xSize);
      if(board.colors[loc] == C_EMPTY)
        continue;
      char direction = directionRows[y][x];
      CrosscutDirection expected = variant == QuaxVariant::DoubleCrosscut ? CROSSCUT_BOTH : direction == '/' ? CROSSCUT_SLASH : CROSSCUT_BACKSLASH;
      if((variant == QuaxVariant::DoubleCrosscut && direction != 'b') ||
         (QuaxVariantIO::hasDirectionalCrosscuts(variant) && direction != '\\' && direction != '/'))
        throw StringError("Board::ofJson - invalid crosscut direction");
      if(expected != board.crosscutDirections[loc]) {
        Color color = board.colors[loc];
        board.setStone(loc,C_EMPTY);
        board.setStone(board.getSecondCrosscutLoc(loc),color);
      }
    }
  }
  return board;
}
