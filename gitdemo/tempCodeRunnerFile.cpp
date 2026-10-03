#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <ctime>
#include <cctype>
#include <random>
#include <map>

using namespace std;

class Hangman {
private:
    const vector<string> HANGMAN_PICS = {
        "  +---+\n  |   |\n      |\n      |\n      |\n      |\n=========",
        "  +---+\n  |   |\n  O   |\n      |\n      |\n      |\n=========",
        "  +---+\n  |   |\n  O   |\n  |   |\n      |\n      |\n=========",
        "  +---+\n  |   |\n  O   |\n /|   |\n      |\n      |\n=========",
        "  +---+\n  |   |\n  O   |\n /|\\  |\n      |\n      |\n=========",
        "  +---+\n  |   |\n  O   |\n /|\\  |\n /    |\n      |\n=========",
        "  +---+\n  |   |\n  O   |\n /|\\  |\n / \\  |\n      |\n========="
    };

    const map<string, vector<string>> WORD_CATEGORIES = {
        {"Animals", {"elephant", "giraffe", "kangaroo", "penguin", "lion", "tiger", "bear", "monkey", "zebra", "hippo", "cheetah", "gorilla", "fox", "wolf"}},
        {"Countries", {"canada", "brazil", "japan", "germany", "france", "italy", "spain", "russia", "china", "india", "australia", "mexico", "egypt", "nigeria"}},
        {"Fruits", {"banana", "apple", "orange", "mango", "strawberry", "grape", "watermelon", "pineapple", "kiwi", "blueberry", "raspberry", "peach", "pear", "plum"}},
        {"Sports", {"football", "basketball", "tennis", "baseball", "golf", "swimming", "boxing", "cricket", "volleyball", "rugby", "hockey", "badminton", "skiing", "cycling"}}
    };

    string secretWord;
    string guessedWord;
    int incorrectGuesses;
    string usedLetters;

public:
    Hangman() : incorrectGuesses(0), usedLetters("") {
        displayHangmanArt();
        string category = chooseCategory();
        secretWord = chooseWord(category);
        if (secretWord.empty()) {
            cout << "Error: Category not found." << endl;
            exit(1);
        }
        guessedWord = string(secretWord.length(), '_');
    }

    void play() {
        while (incorrectGuesses < 6 && guessedWord != secretWord) {
            displayGame();
            char guess = getGuess();
            usedLetters += guess;

            if (secretWord.find(guess) != string::npos) {
                updateGuessedWord(guess);
            } else {
                incorrectGuesses++;
            }
        }

        displayGame();
        displayResult();
    }

private:
    string chooseWord(const string& category) {
        auto it = WORD_CATEGORIES.find(category);
        if (it == WORD_CATEGORIES.end()) {
            return "";
        }
        const vector<string>& words = it->second;
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dist(0, words.size() - 1);
        return words[dist(gen)];
    }

    void displayGame() {
        cout << HANGMAN_PICS[incorrectGuesses] << endl;
        cout << "Word: " << guessedWord << endl;
        cout << "Used letters: " << usedLetters << endl;
        cout << "Incorrect guesses: " << incorrectGuesses << "/6" << endl;
    }

    char getGuess() {
        char guess;
        while (true) {
            cout << "Guess a letter: ";
            cin >> guess;
            guess = tolower(guess);
            if (!isalpha(guess)) {
                cout << "Invalid input. Please enter a letter." << endl;
            } else if (usedLetters.find(guess) != string::npos) {
                cout << "You've already guessed that letter." << endl;
            } else {
                break;
            }
        }
        return guess;
    }

    void updateGuessedWord(char guess) {
        for (size_t i = 0; i < secretWord.length(); ++i) {
            if (secretWord[i] == guess) {
                guessedWord[i] = guess;
            }
        }
    }

    void displayHangmanArt() {
        cout << R"(
 _   _                                         
| | | |                                        
| |_| | __ _ _ __   __ _ _ __ ___   __ _ _ __  
|  _  |/ _` | '_ \ / _` | '_ ` _ \ / _` | '_ \ 
| | | | (_| | | | | (_| | | | | | | (_| | | | |
|_| |_|\__,_|_| |_|\__, |_| |_| |_|\__,_|_| |_|
                     __/ |                    
                    |___/                     
)" << endl;
    }

    string chooseCategory() {
        cout << "Choose a category:" << endl;
        int i = 1;
        for (const auto& category : WORD_CATEGORIES) {
            cout << i++ << ". " << category.first << endl;
        }
        int choice;
        while (true) {
            cout << "Enter your choice: ";
            cin >> choice;
            if (choice >= 1 && choice <= WORD_CATEGORIES.size()) {
                auto it = WORD_CATEGORIES.begin();
                advance(it, choice - 1);
                return it->first;
            } else {
                cout << "Invalid choice." << endl;
            }
        }
    }

    void displayResult() {
        if (guessedWord == secretWord) {
            cout << "\n\n*****************************************\n";
            cout << "  Congratulations! You are a Hangman Master!\n";
            cout << "*****************************************\n\n";
        } else {
            cout << HANGMAN_PICS[6] << endl;
            cout << "\n\n*****************************************\n";
            cout << "  Game Over! Better luck next time.\n";
            cout << "  The word was: " << secretWord << "\n";
            cout << "*****************************************\n\n";
        }
    }
};

int main() {
    Hangman game;
    game.play();
    return 0;
}