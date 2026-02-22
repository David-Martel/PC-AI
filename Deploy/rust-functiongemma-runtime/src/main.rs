use std::net::SocketAddr;
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "Config/pcai-functiongemma.json")]
    config: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    rust_functiongemma_runtime::init_runtime_config(&args.config);
    let addr: SocketAddr = rust_functiongemma_runtime::runtime_addr()?;
    rust_functiongemma_runtime::serve(addr).await
}
