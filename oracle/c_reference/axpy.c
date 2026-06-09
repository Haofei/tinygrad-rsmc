void E_8n4(float* restrict data0_8, float* restrict data1_8, float* restrict data2_8, float* restrict data3_8) {
  for (int Lidx0 = 0; Lidx0 < 8; Lidx0++) {
    float val0 = (*(data1_8+Lidx0));
    float val1 = (*(data2_8+Lidx0));
    float val2 = (*(data3_8+Lidx0));
    *(data0_8+Lidx0) = ((val0*val1)+val2);
  }
}
