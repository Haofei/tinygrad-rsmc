// tinygrad ClangRenderer output, NOOPT=1, for (a+b) on size-8 float tensors.
// Captured 2026-06-09 as the reference "shape" the MC renderer must reproduce.
void E_8(float* restrict data0_8, float* restrict data1_8, float* restrict data2_8) {
  for (int Lidx0 = 0; Lidx0 < 8; Lidx0++) {
    float val0 = (*(data1_8+Lidx0));
    float val1 = (*(data2_8+Lidx0));
    *(data0_8+Lidx0) = (val0+val1);
  }
}
