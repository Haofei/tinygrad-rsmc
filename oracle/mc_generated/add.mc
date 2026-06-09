export fn E_8n1(data0_8: *mut f32, data1_8: *mut f32, data2_8: *mut f32) -> void {
  var Lidx0: usize = 0;
  while Lidx0 < 8 {
    let val0: f32 = data1_8[Lidx0];
    let val1: f32 = data2_8[Lidx0];
    data0_8[Lidx0] = (val0+val1);
    Lidx0 = Lidx0 + 1;
  }
}
