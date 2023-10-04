// ----------------- [variables] ------------------------
fn sample_var() {
  let un_mutable_var = 1;

  // Error, can't change value
  //un_mutable_var = 2;

  let mut mutable_var = 2;
  mutable_var = mutable_var + un_mutable_var;

  println!("mutable_var {}", mutable_var);

  // shadowing the previous variable
  let un_mutable_var = 100;
  println!("un_mutable_var {}", un_mutable_var);
}

// ----------------- [strings] ------------------------
fn fn_string(string_var: String) {
  println!("type:String {}", string_var);
}

fn fn_string_ref(string_ref: &String) {
  println!("type:&String {}", string_ref);
}

fn fn_str(str_var: &str) {
  println!("type:&str {}", str_var);
}

fn sample_str() {
  // Owned object
  let string_var: String = "String var".to_string();

  // Borrow ownership
  fn_string_ref(&string_var);

  // Transfer ownership: string_var cannot be used after the function call.
  fn_string(string_var);

  // Not owned, a slice
  let str_var: &str = "str var";
  fn_str(str_var);
}

// ----------------- [vector] ------------------------
fn sample_vec() {
  let str_vec: Vec<&str> = vec!["abc", "def", "hij"];
  let int_vec: Vec<i32> = vec![1, 2, 3, 4];

  println!("{:?}", str_vec);
  println!("{:?}", int_vec);

  for s in str_vec {
    println!("{}", s);
  }

  for i in 0..int_vec.len() {
    println!("{}", int_vec[i]);
  }
}

fn main() {
  sample_var();
  sample_str();
  sample_vec();
}
