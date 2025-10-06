Dla przypomnienia, zadanie ma teoretycznie 3 etapy:
1) użycie heurystyki (lub teoretycznie modelu ML), aby wykryć jak najwięcej kropek,
2) użycie modelu ML, aby znaleźć transformację. Jak już mamy transformację, to możemy wtedy zmatchować dwie spirale uzyskując m.in. miejsca, gdzie kropki powinny być, a ich nie ma. Kropki może nie być na swoim miejscu z dwóch powodów. Albo heurystyka z pkt 1 jej nie wykryła albo w ogóle nie było tam kropki.
3) ustalenie, co znajduje się w “miejscach na brakujące kropki”

**Ważne:** wszystkie skrypty, jakie są używane w tym zadaniu mają swoje parametry domyślne, 
a ich outputy znajdują się w folderach o nazwach zgodnych ze wzorem _outputs/_spirale/<nazwa_skryptu>.

### Heurystyka do wykrywania kropek
Aby użyć heurystyki z pkt. 1) odpalamy skrypt
```bash
python step1_heurystyka.py
```

Domyślnie jako obraz wejściowy brany jest obrazek _inputs/PXL_20250925_061456317.jpg, zaś 
ostateczna wizualizacja zapisywana jest w _outputs/_spirale/step1_heurystyka/PXL_20250925_061456317.jpg. 
Można to zmienić bawiąc się parametrami skryptu. 

### Model ML-owy do matchowania spirali

Na początek generujemy bazową spiralę Sacksa i zapisujemy ją w pliku .npz
```bash
python step2a_create_canonical.py
```
Współrzędne oryginalnej spirali stanowią dane wejściowe dla kolejncyh skryptów.

Następnie generujemy syntetyczny dataset spiral z różnymi transformacjami. W tym celu odpalamy skrypt
```bash
python step3a_make_dataset.py
```
W katalogu wyjściowym oprócz samego datasetu (.npz) znajdują się również jego wizualizacje (.png). Polecam się
z nimi zapoznać.

Kolejnym etapem jest wytrenowanie modelu ML-owego na danych syntetycznych. W tym celu uruchamiamy
skrypt
```bash
python step3b_train.py
```

W końcu możemy przeprowadzić inferencję za pomocą skryptu
```bash
python step3c_infer.py
```
Domyślnie jako obraz wejściowy brany jest obrazek _inputs/PXL_20250925_061456317_cut_shifted.jpg, zaś 
ostateczna wizualizacja zapisywana jest w _outputs/_spirale/step3c_infer. 
Można to zmienić bawiąc się parametrami skryptu. 

### Heurystyka do matchowania spirali 
Jak się łatwo przekonać model ML-owy nie działa zbyt dobrze. Warto jeszcze sprawdzic, jak dziala 
nie-MLowa heurystyka szukająca transformacji spirali. Aby to zrobić należy uruchomić skrypt
```bash
python step2b_fit_transform.py
```
Domyślnie jako obraz wejściowy brany jest obrazek _inputs/PXL_20250925_061456317_cut_shifted.jpg, zaś 
ostateczna wizualizacja zapisywana jest w _outputs/_spirale/step2b_fit_transform. 
Można to zmienić bawiąc się parametrami skryptu. 

**Ważne:** skrypt step2b_fit_transform.py potrzebuje do swojego działania outputu skryptu step2a_create_canonical.py 