use clipboard_win::{formats, get_clipboard, set_clipboard};
use latex2mmlc::{latex_to_mathml, Display};
use std::io::Write;
use std::path::PathBuf;
use VLBindTest::*;
#[cfg(feature = "vl")]
const PROMPT: &str = r#"OCR the image (if symbol use *latex*) to *content only*,no any others:(output content only,*no* explain or others)"#;
fn main() {
    #[cfg(feature = "vl")]
    {
        // let image_path = r#"D:\whl\Download\demo.png"#;
        // 获取命令行参数
        let args: Vec<String> = std::env::args().collect();

        // 提取参数
        let model = std::path::PathBuf::from(
            &args
                .get(1)
                .unwrap_or(&r#"D:\whl\Download\Qwen2-VL-7B-Instruct-Q4_K_M.gguf"#.to_string()),
        );
        let mmproj = std::path::PathBuf::from(
            &args
                .get(2)
                .unwrap_or(&r#"D:\whl\Download\mmproj-Qwen2-VL-7B-Instruct-f16.gguf"#.to_string()),
        );

        // 检查参数数量
        // if args.len() < 3 {
        //     eprintln!("Usage: {} <model> <mmproj>", args[0]);
        //     return;
        // }
        if !model.exists() {
            eprintln!("{} not exist!", model.display());
        }
        if !mmproj.exists() {
            eprintln!("{} not exist!", mmproj.display());
        }

        let args = vec![
            "vl".to_string(),
            "-m".to_string(),
            // r#"D:\whl\Download\Qwen2-VL-7B-Instruct-Q4_K_M.gguf"#.to_string(),
            model.display().to_string(),
            "--mmproj".to_string(),
            // r#"D:\whl\Download\mmproj-Qwen2-VL-7B-Instruct-f16.gguf"#.to_string(),
            mmproj.display().to_string(),
            "-p".to_string(),
            PROMPT.to_string(),
            "-ngl".to_string(),
            "100".to_string(),
            "--log-verbosity".to_string(),
            "-1".to_string(),
        ];

        let ctx = LlavaContext::new(args).expect("Failed to initialize LlavaContext");

        // let s = std::time::Instant::now();
        // // let image_bytes = vec![0u8; 1024]; // 模拟图像数据
        // let image_bytes = std::fs::read(image_path).expect("cant find image");
        // println!("inject images...");
        // let result = ctx.run(&image_bytes).expect("Failed to run inference");
        //
        // println!("Result: {}", result.get_result());
        // println!("Time cost: {:?}", s.elapsed());

        // 进入交互循环
        loop {
            // 打印提示符
            print!("(image path) > ");
            std::io::stdout().flush().unwrap(); // 确保提示符立即显示

            // 读取用户输入
            let mut input = String::new();
            std::io::stdin()
                .read_line(&mut input)
                .expect("Failed to read line");

            // 去掉输入末尾的换行符
            let input = input.trim();

            // 处理输入
            match input {
                "exit" => {
                    println!("Exiting...");
                    break; // 退出循环
                }
                "reload" => {
                    ctx.reload_context();
                    println!("context reload!");
                }
                _ => {
                    let s = std::time::Instant::now();
                    // let image_bytes = vec![0u8; 1024]; // 模拟图像数据
                    match std::fs::read(input) {
                        Ok(image_bytes) => {
                            println!("read image {input}...");

                            let result = ctx.run(&image_bytes).expect("Failed to run inference");
                            let result = result.get_result();
                            eprintln!("raw:{result}");

                            let result = result
                                .trim()
                                .replace("\\]", "")
                                .replace("\\[", "")
                                .trim_matches('$')
                                .to_string();
                            let mathml = match latex_to_mathml(&result, Display::Inline, false) {
                                Ok(m) => m.replace(
                                    "<math>",
                                    r#"<math xmlns="http://www.w3.org/1998/Math/MathML">"#,
                                ),
                                Err(err) => {
                                    eprintln!("convert mathml fail :{err}");
                                    result
                                }
                            };
                            set_clipboard(formats::Unicode, &mathml).expect("To set clipboard");
                            println!("Result: {}", mathml);
                            ctx.reload_context();
                        }
                        Err(e) => {
                            eprintln!("Error read image:{e}")
                        }
                    }
                    println!("Time cost: {:?}", s.elapsed());
                }
            }
        }
    }

    #[cfg(feature = "got")]
    {
        let args: Vec<String> = std::env::args().collect();
        let encoder_model_path = args
            .get(1)
            .map(|s| std::path::PathBuf::from(s))
            .filter(|p| p.exists())
            .unwrap_or(std::path::PathBuf::from(
                r#".\encoder_single.onnx"#,
            ));
        let decoder_model_path = args
            .get(1)
            .map(|s| std::path::PathBuf::from(s))
            .filter(|p| p.exists())
            .unwrap_or(std::path::PathBuf::from(
                r#".\got_decoder-q4_k_m.gguf"#,
            ));

        let s = std::time::Instant::now();
        let model = GotOcr::from_encoder_decoder(encoder_model_path, decoder_model_path).unwrap();
        println!("Load model time cost: {:?}", s.elapsed());

        // 进入交互循环
        loop {
            // 打印提示符
            print!("(image path) > ");
            std::io::stdout().flush().unwrap(); // 确保提示符立即显示

            // 读取用户输入
            let mut input = String::new();
            std::io::stdin()
                .read_line(&mut input)
                .expect("Failed to read line");

            // 去掉输入末尾的换行符
            let input = input.trim();

            // 处理输入
            match input {
                "exit" => {
                    println!("Exiting...");
                    break; // 退出循环
                }
                "reload" => {
                    model.cleanup_ctx().unwrap();
                    println!("context reload!");
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
                    match model.generate(input, GotType::Format) {
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
