import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the improved model and preprocessors
def load_improved_model():
    model = tf.keras.models.load_model('ann_model_improved_final.h5')
    
    with open("scaler_improved.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    with open("label_encoder-output.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    # Load all other encoders
    encoders = {}
    encoder_files = [
        "le_spf_result", "le_request_type", "le_dkim_result",
        "le_dmarc_result", "le_tls_version", "le_ssl_validity_status",
        "le_unique_parent_process_names"
    ]
    
    for encoder_name in encoder_files:
        with open(f"{encoder_name}.pkl", "rb") as f:
            encoders[encoder_name] = pickle.load(f)
    
    return model, scaler, label_encoder, encoders

# Feature names list
feature_names = [
    "sender_known_malicious",
    "sender_domain_reputation_score",
    "sender_spoof_detected",
    "sender_temp_email_likelihood",
    "dmarc_enforced",
    "packer_detected",
    "any_file_hash_malicious",
    "max_metadata_suspicious_score",
    "malicious_attachment_count",
    "has_executable_attachment",
    "unscannable_attachment_present",
    "total_yara_match_count",
    "total_ioc_count",
    "max_behavioral_sandbox_score",
    "max_amsi_suspicion_score",
    "any_macro_enabled_document",
    "any_vbscript_javascript_detected",
    "any_active_x_objects_detected",
    "any_network_call_on_open",
    "max_exfiltration_behavior_score",
    "any_exploit_pattern_detected",
    "total_embedded_file_count",
    "max_suspicious_string_entropy_score",
    "max_sandbox_execution_time",
    "unique_parent_process_names",
    "return_path_mismatch_with_from",
    "return_path_known_malicious",
    "return_path_reputation_score",
    "reply_path_known_malicious",
    "reply_path_diff_from_sender",
    "reply_path_reputation_score",
    "smtp_ip_known_malicious",
    "smtp_ip_geo",
    "smtp_ip_asn",
    "smtp_ip_reputation_score",
    "domain_known_malicious",
    "url_count",
    "dns_morphing_detected",
    "domain_tech_stack_match_score",
    "is_high_risk_role_targeted",
    "sender_name_similarity_to_vip",
    "urgency_keywords_present",
    "request_type",
    "content_spam_score",
    "user_marked_as_spam_before",
    "bulk_message_indicator",
    "unsubscribe_link_present",
    "marketing-keywords_detected",
    "html_text_ratio",
    "image_only_email",
    "spf_result",
    "dkim_result",
    "dmarc_result",
    "reverse_dns_valid",
    "tls_version",
    "total_links_detected",
    "url_shortener_detected",
    "url_redirect_chain_length",
    "final_url_known_malicious",
    "url_decoded_spoof_detected",
    "url_reputation_score",
    "ssl_validity_status",
    "site_visual_similarity_to_known_brand",
    "url_rendering_behavior_score",
    "link_rewritten_through_redirector",
    "token_validation_success",
    "total_components_detected_malicious",
    "Analysis_of_the_qrcode_if_present",
]

# Default values
default_values = {
    "sender_known_malicious": 0,
    "sender_domain_reputation_score": 0.95,
    "sender_spoof_detected": 0,
    "sender_temp_email_likelihood": 0.0,
    "dmarc_enforced": 1,
    "packer_detected": 0,
    "any_file_hash_malicious": 0,
    "max_metadata_suspicious_score": 0.0,
    "malicious_attachment_count": 0,
    "has_executable_attachment": 0,
    "unscannable_attachment_present": 0,
    "total_yara_match_count": 0,
    "total_ioc_count": 0,
    "max_behavioral_sandbox_score": 0.0,
    "max_amsi_suspicion_score": 0.0,
    "any_macro_enabled_document": 0,
    "any_vbscript_javascript_detected": 0,
    "any_active_x_objects_detected": 0,
    "any_network_call_on_open": 0,
    "max_exfiltration_behavior_score": 0.0,
    "any_exploit_pattern_detected": 0,
    "total_embedded_file_count": 0,
    "max_suspicious_string_entropy_score": 0.075607,
    "max_sandbox_execution_time": 4.377776483e-107,
    "unique_parent_process_names": '[""]',
    "return_path_mismatch_with_from": 0,
    "return_path_known_malicious": 0,
    "return_path_reputation_score": 0.95,
    "reply_path_known_malicious": 0,
    "reply_path_diff_from_sender": 0,
    "reply_path_reputation_score": 0.95,
    "smtp_ip_known_malicious": 0,
    "smtp_ip_geo": 0.001,
    "smtp_ip_asn": 0.05,
    "smtp_ip_reputation_score": 0.95,
    "domain_known_malicious": 0,
    "url_count": 0,
    "dns_morphing_detected": 0,
    "domain_tech_stack_match_score": 1.0,
    "is_high_risk_role_targeted": 0,
    "sender_name_similarity_to_vip": 0.0,
    "urgency_keywords_present": 0,
    "request_type": "none",
    "content_spam_score": 0.0,
    "user_marked_as_spam_before": 0,
    "bulk_message_indicator": 0,
    "unsubscribe_link_present": 0,
    "marketing-keywords_detected": 0,
    "html_text_ratio": 0.0,
    "image_only_email": 0,
    "spf_result": "pass",
    "dkim_result": "pass",
    "dmarc_result": "pass",
    "reverse_dns_valid": 1,
    "tls_version": "TLS 1.2",
    "total_links_detected": 0,
    "url_shortener_detected": 0,
    "url_redirect_chain_length": 0,
    "final_url_known_malicious": 0,
    "url_decoded_spoof_detected": 0,
    "url_reputation_score": 0.0,
    "ssl_validity_status": "valid",
    "site_visual_similarity_to_known_brand": 0.0,
    "url_rendering_behavior_score": 0.0000430629,
    "link_rewritten_through_redirector": 0,
    "token_validation_success": 1,
    "total_components_detected_malicious": 0,
    "Analysis_of_the_qrcode_if_present": 2,
}

def predict_improved(input_data, model=None, scaler=None, label_encoder=None, encoders=None):
    """
    Improved prediction function with better preprocessing
    """
    # Load model if not provided
    if model is None:
        model, scaler, label_encoder, encoders = load_improved_model()
    
    data_dict = input_data["data"]
    
    # Create DataFrame
    df = pd.DataFrame([data_dict])
    
    # Fill missing values
    for col in feature_names:
        if col not in df.columns:
            df[col] = default_values.get(col, 0)
    
    # Handle string to boolean conversion
    for col in df.columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if isinstance(x, str) and x.lower() == "true" else
                                              0 if isinstance(x, str) and x.lower() == "false" else x)
    
    # Apply categorical encodings
    categorical_mappings = {
        "request_type": ("le_request_type", 10),  # 10 = none
        "spf_result": ("le_spf_result", 2),       # 2 = none
        "dkim_result": ("le_dkim_result", 2),     # 2 = none
        "dmarc_result": ("le_dmarc_result", 2),   # 2 = none
        "tls_version": ("le_tls_version", 4),     # 4 = TLS 1.2
        "ssl_validity_status": ("le_ssl_validity_status", 8),  # 8 = valid
        "unique_parent_process_names": ("le_unique_parent_process_names", 0)  # 0 = [""]
    }
    
    for col, (encoder_name, default_encoded) in categorical_mappings.items():
        try:
            df[col] = encoders[encoder_name].transform(df[col])
        except:
            df[col] = default_encoded
    
    # Extract features in correct order
    X = df[feature_names].values
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    probabilities = model.predict(X_scaled)
    prediction_class = np.argmax(probabilities, axis=1)
    
    # Get class name
    predicted_label = label_encoder.inverse_transform(prediction_class)[0]
    
    # Get confidence scores
    confidence = {
        label_encoder.classes_[i]: float(probabilities[0][i]) 
        for i in range(len(label_encoder.classes_))
    }
    
    return {
        "prediction": predicted_label,
        "confidence": confidence,
        "max_confidence": float(np.max(probabilities))
    }

def process_excel_improved(excel_file):
    """
    Process Excel file with improved model
    """
    # Load model once
    model, scaler, label_encoder, encoders = load_improved_model()
    
    # Read CSV/Excel
    if excel_file.endswith('.csv'):
        df = pd.read_csv(excel_file)
    else:
        df = pd.read_excel(excel_file)
    
    print(f"Processing {len(df)} rows with improved model...")
    
    results = []
    
    for idx, row in df.iterrows():
        # Convert to input format
        json_input = {"data": row.to_dict()}
        
        # Get prediction
        result = predict_improved(json_input, model, scaler, label_encoder, encoders)
        
        # Save row + prediction
        row_result = row.to_dict()
        row_result['prediction'] = result['prediction']
        row_result['confidence_malicious'] = result['confidence'].get('Malicious', 0)
        row_result['confidence_no_action'] = result['confidence'].get('No Action', 0)
        row_result['confidence_spam'] = result['confidence'].get('Spam', 0)
        row_result['max_confidence'] = result['max_confidence']
        results.append(row_result)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} rows...")
    
    # Save results
    output_file = excel_file.replace('.xlsx', '_predictions_improved.csv').replace('.csv', '_predictions_improved.csv')
    pd.DataFrame(results).to_csv(output_file, index=False)
    
    print(f"\nSaved predictions to: {output_file}")
    
    # Print summary
    predictions_df = pd.DataFrame(results)
    print("\nPrediction Summary:")
    print(predictions_df['prediction'].value_counts())
    
    return output_file

if __name__ == "__main__":
    # Test the improved prediction function
    test_input = {
        "data": {
            "sender_known_malicious": 0,
            "spf_result": "pass",
            "dkim_result": "pass",
            "dmarc_result": "pass",
            "html_text_ratio": 0.5,
            "url_count": 2
        }
    }
    
    result = predict_improved(test_input)
    print(f"Test prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")