# LSTM char-based model
LSTM model is trained to predict the next character based on the given sequence of characters. Theoretically model can be trained for any language, but verification is required.  

### Train model on the given text
`python3 char_model.py --mode=train --name=english_news --text-file=merged_text.txt --seed-len=40 --epochs=10`

### Use pretrained model
`python3 char_model.py --mode=use --name=english_news --seed='Comments Washington Battles' --maxlen=50`