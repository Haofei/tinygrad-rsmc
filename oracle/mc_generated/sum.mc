export fn r_8n1(data0_1: *mut f32, data1_8: *mut f32) -> void {
  var acc0: [1]f32;
  acc0[0] = 0.0;
  var Ridx0: usize = 0;
  while Ridx0 < 8 {
    let val0: f32 = data1_8[Ridx0];
    acc0[0] = (acc0[0]+val0);
    Ridx0 = Ridx0 + 1;
  }
  data0_1[0] = acc0[0];
}
