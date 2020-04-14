use whirl::running::experiments;

fn main() {
    let mut simulation = experiments::rarefaction_wave(1.0, 1.0, 2.0, 200, 1.4)
        .unwrap()
        .build()
        .unwrap();

    let time_step = 1e-4;
    for _ in 0..10 {
        println!("{:?}", simulation.positions());
        simulation.step(time_step);
    }
}
