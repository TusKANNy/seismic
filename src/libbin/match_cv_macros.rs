#[macro_export]
macro_rules! match_component_value {
    ($component:expr, $value:expr, $macro_name:ident) => {
        use half::bf16;
        use half::f16;
        use seismic::FixedU8Q;
        use seismic::FixedU16Q;

        match ($component, $value) {
            ("u16", "f16") => {
                $macro_name!(u16, f16);
            }
            ("u16", "bf16") => {
                $macro_name!(u16, bf16);
            }
            ("u16", "f32") => {
                $macro_name!(u16, f32);
            }
            ("u16", "fixedu8") => {
                $macro_name!(u16, FixedU8Q);
            }
            ("u16", "fixedu16") => {
                $macro_name!(u16, FixedU16Q);
            }
            ("u32", "f16") => {
                $macro_name!(u32, f16);
            }
            ("u32", "bf16") => {
                $macro_name!(u32, bf16);
            }
            ("u32", "f32") => {
                $macro_name!(u32, f32);
            }
            ("u32", "fixedu8") => {
                $macro_name!(u32, FixedU8Q);
            }
            ("u32", "fixedu16") => {
                $macro_name!(u32, FixedU16Q);
            }
            _ => {
                eprintln!(
                    "Error: component-type must be either 'u16' or 'u32', value-type must be 'f16', 'bf16', 'f32', 'fixedu16', or 'fixedu8'"
                );
                std::process::exit(1);
            }
        }
    };
}

#[macro_export]
macro_rules! match_component {
    ($component:expr, $macro_name:ident) => {
        match $component {
            "u16" => {
                $macro_name!(u16);
            }
            "u32" => {
                $macro_name!(u32);
            }
            _ => {
                eprintln!("Error: component-type must be either 'u16' or 'u32'");
                std::process::exit(1);
            }
        }
    };
}

#[macro_export]
macro_rules! match_value {
    ($value:expr, $macro_name:ident) => {
        use half::bf16;
        use half::f16;
        use seismic::FixedU8Q;
        use seismic::FixedU16Q;

        match $value {
            "f16" => {
                $macro_name!(f16);
            }
            "bf16" => {
                $macro_name!(bf16);
            }
            "f32" => {
                $macro_name!(f32);
            }
            "fixedu8" => {
                $macro_name!(FixedU8Q);
            }
            "fixedu16" => {
                $macro_name!(FixedU16Q);
            }
            _ => {
                eprintln!(
                    "Error: value-type must be 'f16', 'bf16', 'f32', 'fixedu16', or 'fixedu8'"
                );
                std::process::exit(1);
            }
        }
    };
}
