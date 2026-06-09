// tinygrad ClangRenderer output, NOOPT=1, for a.sum() on a size-8 float tensor.
// Captured 2026-06-09. Note the reduce accumulator pattern (acc array of size 1).
void r_8(float* restrict data0_1, float* restrict data1_8) {
  float acc0[1];
  *(acc0+0) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 8; Ridx0++) {
    float val0 = (*(data1_8+Ridx0));
    *(acc0+0) = ((*(acc0+0))+val0);
  }
  *(data0_1+0) = (*(acc0+0));
}
