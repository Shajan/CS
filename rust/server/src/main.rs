extern crate warp;
extern crate tokio;
extern crate serde;
extern crate serde_json;

use warp::Filter;
use serde_json::Value;

#[tokio::main]
async fn main() {
    // Define a GET /hello/warp route
    let hello = warp::path!("hello" / "warp")
        .map(|| warp::reply::html("Hello, warp!"));

    // Define a POST /echo route
    let echo = warp::post()
        .and(warp::path("echo"))
        .and(warp::body::json())
        .map(|body: Value| {
            warp::reply::json(&body)
        });

    // Combine the above routes
    let routes = hello.or(echo);

    // Start the warp server
    warp::serve(routes).run(([127, 0, 0, 1], 8080)).await;
}

