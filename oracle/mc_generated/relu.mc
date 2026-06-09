export fn E_8n7(data0_8: *mut f32, data1_8: *mut f32) -> void {
  var Lidx0: usize = 0;
  while Lidx0 < 8 {
    let val0: f32 = data1_8[Lidx0];
    let alu0: f32 = (if (0.0<val0) { val0 } else { 0.0 });
    data0_8[Lidx0] = alu0;
    Lidx0 = Lidx0 + 1;
  }
}
