use clipboard_win::{formats, get_clipboard, set_clipboard};
use got_rs::*;
use latex2mmlc::{latex_to_mathml, Display};
use std::io::Write;
use std::path::PathBuf;
use std::process::exit;

fn main() {
    let args = parse_args();
    let s = std::time::Instant::now();
    let model = GotOcr::from_encoder_decoder(&args.encoder, &args.decoder)
        .expect("cant find model in path!");
    println!("Load model time cost: {:?}", s.elapsed());
    let mode = args.mode;
    let mut ocr_type = args.r#type;
    match mode {
        Mode::Once => {
            // panic!("Not implement yet...")
            match args.image_path {
                None => {
                    println!("no image");
                    exit(-1);
                }
                Some(path) => {
                    let s = std::time::Instant::now();
                    match model.generate(path, ocr_type.into()) {
                        Ok(result) => {
                            // let result = ctx.run(&image_bytes).expect("Failed to run inference");
                            // let result = result;
                            eprintln!("Raw result: \x1b[31m <* {result} *> \x1b[0m ");

                            match result.get_result() {
                                Ok(s) => {
                                    let result = s
                                        .trim()
                                        .replace("\\]", "")
                                        .replace("\\[", "")
                                        .replace("\\)", "")
                                        .replace("\\(", "")
                                        .trim_matches('$')
                                        .to_string();
                                    let mathml = match latex_to_mathml(
                                        &result,
                                        Display::Inline,
                                        false,
                                    ) {
                                        Ok(m) => m.replace(
                                            "<math>",
                                            r#"<math xmlns="http://www.w3.org/1998/Math/MathML">"#,
                                        ),
                                        Err(err) => {
                                            eprintln!("convert mathml fail :{err}");
                                            result
                                        }
                                    };
                                    set_clipboard(formats::Unicode, &mathml)
                                        .expect("To set clipboard");
                                    println!("MathMl:\x1b[034m {} \x1b[0m", mathml);
                                }
                                Err(e) => {
                                    eprintln!("Error while ocr:{e}");
                                }
                            };
                        }
                        Err(e) => {
                            eprintln!("Error while ocr:{e}")
                        }
                    }
                    println!("Time cost: {:?}", s.elapsed());
                }
            };
        }
        Mode::Chat => {
            loop {
                const PROMPT: &str = "image path";
                print!("({PROMPT}) > ");
                std::io::stdout().flush().unwrap();

                let mut input = String::new();
                std::io::stdin()
                    .read_line(&mut input)
                    .expect("Failed to read line");

                let input = input.trim();

                match input {
                    "exit" => {
                        println!("Exiting...");
                        break;
                    }

                    "reload" => {
                        model.cleanup_ctx().unwrap();
                        println!("context reload!");
                    }
                    s if s.starts_with("type") => {
                        let _ = s.split(" ").collect::<Vec<_>>().get(1).map(|t| {
                            ocr_type = match t.to_lowercase().as_str() {
                                "ocr" => OcrType::Ocr,
                                "format" => OcrType::Format,
                                "crop" => OcrType::CropOcr,
                                "crop_format" => OcrType::CropFormat,
                                _ => OcrType::Ocr,
                            };
                            println!("change ocr type to {ocr_type}");
                        });
                    }
                    _ => {
                        let s = std::time::Instant::now();

                        // if PathBuf::from(input).exists() {
                        //     println!("read image {input}...");
                        // }else {
                        //     println!("path not exist ,use clipboard...");
                        //     match get_clipboard(formats::Bitmap)     {
                        //         Ok(img)=>{
                        //
                        //         }
                        //         Err(_e)=>{
                        //             println!("no images in clipboard...");
                        //             continue
                        //         }
                        //     }
                        // }
                        match model.generate(input, ocr_type.into()) {
                            Ok(result) => {
                                // let result = ctx.run(&image_bytes).expect("Failed to run inference");
                                // let result = result;
                                eprintln!("Raw result: \x1b[31m <* {result} *> \x1b[0m ");

                                match result.get_result() {
                                    Ok(s) => {
                                        let result = s
                                            .trim()
                                            .replace("\\]", "")
                                            .replace("\\[", "")
                                            .replace("\\)", "")
                                            .replace("\\(", "")
                                            .trim_matches('$')
                                            .to_string();
                                        let mathml = match latex_to_mathml(
                                            &result,
                                            Display::Inline,
                                            false,
                                        ) {
                                            Ok(m) => m.replace(
                                                "<math>",
                                                r#"<math xmlns="http://www.w3.org/1998/Math/MathML">"#,
                                            ),
                                            Err(err) => {
                                                eprintln!("convert mathml fail :{err}");
                                                result
                                            }
                                        };
                                        set_clipboard(formats::Unicode, &mathml)
                                            .expect("To set clipboard");
                                        println!("MathMl:\x1b[034m {} \x1b[0m", mathml);
                                    }
                                    Err(e) => {
                                        eprintln!("Error while ocr:{e}");
                                    }
                                };
                            }
                            Err(e) => {
                                eprintln!("Error while ocr:{e}")
                            }
                        }
                        println!("Time cost: {:?}", s.elapsed());
                    }
                }
            }
        }
    }
}
