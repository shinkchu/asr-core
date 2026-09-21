use super::*;

#[test]
fn websocket_transport_errors_keep_their_source() {
    let io = std::io::Error::new(std::io::ErrorKind::ConnectionReset, "reset");
    let error = websocket_error("receive", tungstenite::Error::Io(io));
    assert_eq!(error.kind, ErrorKind::Io);
    assert_eq!(error.stage, "receive");
}

#[test]
fn server_error_composition_degrades_and_caps_fields() {
    use serde_json::json;

    // code 与 message 都可渲染：组装为 "prefix (code: message)"。
    let full = server_error("prefix", &json!("quota"), &json!("boom"));
    assert_eq!(full.kind, ErrorKind::Protocol);
    assert_eq!(full.message, "prefix (quota: boom)");

    // 只缺 code 退化为 "prefix (message)"；message 缺失（含空白）退回
    // 静态前缀，code 单独存在不渲染。
    let message_only = server_error("prefix", &Value::Null, &json!("boom"));
    assert_eq!(message_only.message, "prefix (boom)");
    let bare_code = server_error("prefix", &json!("quota"), &Value::Null);
    assert_eq!(bare_code.message, "prefix");
    let neither = server_error("prefix", &Value::Null, &Value::Null);
    assert_eq!(neither.message, "prefix");
    let blank = server_error("prefix", &json!("   "), &json!("boom"));
    assert_eq!(blank.message, "prefix (boom)");

    // 超长字段截断到上限并加省略号。
    let oversized = server_error("prefix", &Value::Null, &json!("y".repeat(1000)));
    assert_eq!(
        oversized.message,
        format!("prefix ({}…)", "y".repeat(SERVER_ERROR_FIELD_LIMIT))
    );
}
