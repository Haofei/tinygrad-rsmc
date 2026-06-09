void r_8(float* restrict data0_1, float* restrict data1_8) {
  float acc0[1];
  *(acc0+0) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 8; Ridx0++) {
    float val0 = (*(data1_8+Ridx0));
    *(acc0+0) = ((*(acc0+0))+val0);
  }
  *(data0_1+0) = (*(acc0+0));
}
