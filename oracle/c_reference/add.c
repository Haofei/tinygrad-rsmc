void E_8(float* restrict data0_8, float* restrict data1_8, float* restrict data2_8) {
  for (int Lidx0 = 0; Lidx0 < 8; Lidx0++) {
    float val0 = (*(data1_8+Lidx0));
    float val1 = (*(data2_8+Lidx0));
    *(data0_8+Lidx0) = (val0+val1);
  }
}
