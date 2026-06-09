import "std/addr.mc";

fn select_f32(c: bool, a: f32, b: f32) -> f32 {
  if c { return a; } else { return b; }
}

export fn E_8n7(data0_8: PAddr, data1_8: PAddr) -> void {
  unsafe {
    var Lidx0: usize = 0;
    while Lidx0 < 8 {
      let val0: f32 = raw.load<f32>(pa_offset(data1_8, (Lidx0) * 4));
      let alu0: f32 = select_f32((0.0<val0), val0, 0.0);
      raw.store<f32>(pa_offset(data0_8, (Lidx0) * 4), alu0);
      Lidx0 = Lidx0 + 1;
    }
  }
}
