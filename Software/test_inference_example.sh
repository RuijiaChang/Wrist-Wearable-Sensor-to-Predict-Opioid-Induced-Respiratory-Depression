python Software/infer.py \
  --current_spo2_model Software/Checkpoints/current_spo2_model.pkl \
  --current_rr_model Software/Checkpoints/current_rr_model.pkl \
  --future_model Software/Checkpoints/transformer_xgboost_model_extended_10-300s.pkl \
  --input_json Software/Test_Infer/test_input_features.json \
  --output Software/Test_Infer/test_prediction_output.json

