# Model do analizy sentymentu wykonany w Pythonie

Celem projektu jest zaprezentowanie modelu sieci neuronowej służącego do analizy sentymentu recenzji filmowych w języku angielskim. 

## Źródło danych

Dane wykorzystane zarówno do uczenia jak i do walidacji zostały zaczerpnięte z projektu pod adresem [https://nlp.stanford.edu/sentiment/index.html](https://nlp.stanford.edu/sentiment/index.html), który jest rezultatem pracy nad pracą naukową *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank* [[dostępnej tutaj]](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf). Składa się on z 12 000 fragmentów recenzji wraz z klasyfikacją odpowiadającego im sentymentu w skali od 0 do 1 włącznie. Im bliżej sentyment jest wartości 1, tym bardziej pochlebna jest wypowiedź.

## Zastosowany model
W celu stworzenia modelu wykorzystano bibliotekę Keras wykorzystującą TensorFlow. Wykorzystano również bibliotekę NumPy, aby przerobić słowa na wektory, a konkretne zdania na trójwymiarowe tensory (mogłyby one być macierzami, ale API Keras wymagało na wejściu wspomnianego tensora).

Model składa się 

- Warstwy typu *Dense* z funkcją aktywacji - *tangens hiperboliczny*, która kompresuje każde słowo do wektora 50 elementów.
- 3 warstw rekurencyjnych, dwukierunkowych, każda składająca się z 300 komórek GRU (Gated recurrent unit). Dwukierunkowość polega na tym, że na wyjściu dokonywana jest konkatenacja.
- Kolejnych dwóch warstw typu *Dense* mających na celu spłaszczyć wynik do postaci jednej wartości.

Jako funkcję kosztu zastosowano błąd średniokwadratowy.

Model zapisywany był co epochę, których liczbę ustalono na 30.  

## Przykładowe estymaty

"Very bad movie": 0.17
"Bad movie":  0.2
"Good movie": 0.79
"Very good movie": 0.83

"It was very disappointing": 0.16
"It was ok, but i have seen better productions": 0.54
"Below average": 0.41

## Jak uruchomić?

Środowisko virtualenv niezbędne do uruchomienia skryptu można utowrzyć przy pomocy pliku requirements.txt znajdującego się w repozytorium.

W celu sprawdzenia sentymentu wybranego zdania w języku angielskim należy uruchomić skrypt *main.py* i podążać za instrukcjami na ekranie.

