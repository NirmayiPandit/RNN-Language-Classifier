import os
import model
import dataset
import numpy as np

def word_to_hash(chars, word):
    word = word.lower()
    chars = list(chars)
    hashed = [chars.index(char) for char in word]
    while len(hashed) < 10:
        hashed.append(-1)
    return np.ndarray((1, 10), buffer=np.array(hashed), dtype=int)

def get_predicted_language(probs):
    languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
    max_index = np.argmax(probs)
    return (probs[max_index], languages[max_index])

def main():
    language_classifier = model.LanguageClassificationModel()
    data = dataset.LanguageClassificationDataset(language_classifier)
    chars = data.chars

    # Check if all model files exist
    prefix = 'model'
    model_files = [
        f'{prefix}_w.npy',
        f'{prefix}_w_h1.npy',
        f'{prefix}_w_h2.npy',
        f'{prefix}_w_f.npy'
    ]
    all_exist = all(os.path.exists(f) for f in model_files)

    if all_exist:
        language_classifier.load()
        print("Loaded pre-trained model!")
    else:
        print("Training new model...")
        language_classifier.train(data)
        language_classifier.save()
        print("Model saved after training!")

    # Test accuracy
    test_predicted_probs, test_predicted, test_correct = data._predict('test')
    test_accuracy = np.mean(test_predicted == test_correct)
    print(f"\nTest set accuracy: {test_accuracy:.2%}\n")

    # Prediction loop
    while True:
        word = input("Enter a word (press 'q' to quit): ").strip()
        if word.lower() == 'q':
            break
        
        xs = data._encode(word_to_hash(chars, word), None, True)
        result = language_classifier.run(xs)
        probs = data._softmax(result.data)
        max_prob, pred_lang = get_predicted_language(probs[0])
        print(f"Predicted language: {pred_lang} ({max_prob:.2%} confidence)\n")

if __name__ == "__main__":
    main()
