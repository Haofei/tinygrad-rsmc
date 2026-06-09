import "std/addr.mc";

export fn E_8n5(data0_8: PAddr, data1_8: PAddr, data2_8: PAddr, data3_8: PAddr) -> void {
  unsafe {
    var Lidx0: usize = 0;
    while Lidx0 < 8 {
      let val0: f32 = raw.load<f32>(pa_offset(data1_8, (Lidx0) * 4));
      let val1: f32 = raw.load<f32>(pa_offset(data2_8, (Lidx0) * 4));
      let val2: f32 = raw.load<f32>(pa_offset(data3_8, (Lidx0) * 4));
      raw.store<f32>(pa_offset(data0_8, (Lidx0) * 4), ((val0*val1)+val2));
      Lidx0 = Lidx0 + 1;
    }
  }
}
