void E_8n6(float* restrict data0_8, float* restrict data1_8) {
  for (int Lidx0 = 0; Lidx0 < 8; Lidx0++) {
    float val0 = (*(data1_8+Lidx0));
    float alu0 = ((0.0f<val0)?val0:0.0f);
    *(data0_8+Lidx0) = alu0;
  }
}
