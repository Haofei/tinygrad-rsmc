import "std/addr.mc";

export fn r_8n1(data0_1: PAddr, data1_8: PAddr) -> void {
  unsafe {
    var acc0: [1]f32 = uninit;
    acc0[0] = 0.0;
    var Ridx0: usize = 0;
    while Ridx0 < 8 {
      let val0: f32 = raw.load<f32>(pa_offset(data1_8, (Ridx0) * 4));
      acc0[0] = (acc0[0]+val0);
      Ridx0 = Ridx0 + 1;
    }
    raw.store<f32>(pa_offset(data0_1, (0) * 4), acc0[0]);
  }
}
