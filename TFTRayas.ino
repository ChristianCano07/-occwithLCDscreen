#include "SPI.h"
#include "TFT_22_ILI9225.h"

#define TFT_RST A4
#define TFT_RS  A3
#define TFT_CS  A5  // SS
#define TFT_SDI A2  // MOSI
#define TFT_CLK A1  // SCK
#define TFT_LED 0   // 0 if wired to +5V directly

#define TFT_BRIGHTNESS 1
TFT_22_ILI9225 tft = TFT_22_ILI9225(TFT_RST, TFT_RS, TFT_CS, TFT_SDI, TFT_CLK, TFT_LED);
void draw(uint8_t bitmap[],int16_t w, int16_t h){
  tft.drawBitmap(0, 0, bitmap, w,h,COLOR_BLACK,COLOR_WHITE); 
}
  uint8_t aBitmap[5100];


void setup() {
  // put your setup code here, to run once:
tft.begin();
}

void loop() {

for(int i = 0; i < sizeof(aBitmap); i++){;
 //aBitmap[i] = 0xDAf0;
 aBitmap[i] = 0xCDF7;
}
while (true){
  tft.clear();
  draw(aBitmap,180,520);
  delay(1000*60);
}
}
