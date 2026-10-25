import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# Load cleaned NBP communications
df = pd.read_csv("data/nbp_communications.csv")

print("Number of communications:", len(df))
print(df[["date", "clean_text"]].head(3))



hawkish_words = [
    r"wzrost\w*(?:\s+\S+){0,3}?\s+inflacj\w*",
    r"\binflacj\w*(?:\s+\S+){0,8}?\s+wzrosł\w*",
    r"nisk\w* bezroboci\w*",
    r"wysok\w* liczb\w* pracując\w*",
    r"dodatni\w* dynamik\w* sprzedaż\w*",
    r"dodatni\w* dynamik\w* produkcj\w*",
    r"podwyższon\w* dynamik\w* cen\w* usług",
    r"podwyższon\w* roczn\w* dynamik\w* cen\w*",
    r"podwyższon\w* poziom\w* inflacj\w*",
    r"wciąż podwyższon\w* inflacj\w*",
    r"wzrost\w* administrowan\w* cen\w*",
    r"wzrost\w* cen\w* nośnik\w* energi\w*",
    r"wzrost\w* presj\w* popytow\w*",
    r"nasileni\w* presj\w* popytow\w*",
    r"przyspieszeni\w* wzrostu wynagrodze\w*",
    r"wyższ\w* prognoz\w* inflacj\w*",
    r"rosnąc\w* inflacj\w*",
    r"przyspieszeni\w* inflacj\w*",
    r"wysok\w* inflacj\w*",
    r"podwyższon\w* inflacj\w*",
    r"inflacj\w* powyżej celu",
    r"utrzym\w* się wysok\w* inflacj\w*",
    r"utrwal\w* się inflacj\w*",
    r"ryzyko wzrostu inflacj\w*",
    r"ryzyko utrwalenia inflacj\w*",
    r"wzrost\w* presj\w* inflacyjn\w*",
    r"rosnąc\w* presj\w* inflacyjn\w*",
    r"nasil\w* presj\w* inflacyjn\w*",
    r"wzrost\w* oczekiwań inflacyjn\w*",
    r"podwyższon\w* oczekiwani\w* inflacyjn\w*",
    r"wzrost\w*(?:\s+\S+){0,3}?\s+cen\w*",
    r"przyspieszeni\w* dynamik\w* cen\w*",
    r"wzrost\w* cen\w* ropy\w*",
    r"wzrost\w* cen\w* surowc\w*",
    r"wzrost\w*(?:\s+\S+){0,3}?\s+wynagrodze\w*",
    r"przyspieszeni\w* dynamik\w* płac\w*",
    r"rosnąc\w* płac\w*",
    r"presj\w* płacow\w*",
    r"nasil\w* presj\w* płacow\w*",
    r"wzrost\w* zatrudnieni\w*",
    r"popraw\w* sytuacj\w* na rynku pracy",
    r"spadek\w* bezroboci\w*",
    r"obniż\w* stop\w* bezroboci\w*",
    r"wzrost\w* popyt\w* na prac\w*",
    r"wysok\w* wykorzystani\w* zasob\w* pracy",
    r"efekt\w* drugiej rundy",
    r"wzrost\w* gospodarcz\w*",
    r"przyspieszeni\w* wzrostu gospodarcz\w*",
    r"popraw\w* koniunktur\w*",
    r"popraw\w* nastroj\w*",
    r"poprawił\w* się wskaźnik\w* koniunktur\w*",
    r"wzrost\w* produkcj\w*",
    r"wzrost\w* sprzedaż\w*",
    r"wysok\w* tempo wzrostu",
    r"wzrost\w* inwestycj\w*",
    r"wzrost\w* popyt\w* krajow\w*",
    r"wzrost\w* konsumpcj\w*",
    r"wzrost\w* zamówie\w*",
    r"wzrost\w* wykorzystani\w* mocy wytwórcz\w*",
    r"popraw\w* perspektyw\w* wzrostu",
    r"siln\w* aktywnoś\w* gospodarcz\w*",
    r"deprecjacj\w* złot\w*",
    r"osłabieni\w*(?:\s+\S+){0,2}?\s+złot\w*",
    r"wzrost\w* cen\w* import\w*",
    r"wzrost\w* cen\w* energi\w*",
    r"restrykcyjn\w* nastawieni\w*",
    r"utrzym\w* restrykcyjn\w* nastawieni\w*",
    r"zaostrzeni\w* polityk\w* pieniężn\w*",
    r"zacieśnieni\w* polityk\w* pieniężn\w*",
    r"podwyż\w* stóp procentow\w*",
    r"wzrost\w* stóp procentow\w*",
    r"koniecznoś\w* podwyż\w* stóp",
    r"ryzyko dla stabilności cen",
    r"ryzyko przekroczenia celu inflacyjn\w*",
    r"cen\w* ropy\w* wzrosł\w*",
    r"polepszeni\w* sytuacj\w* gospodarc\w*",
    r"polepszył\w* się perspektyw\w* wzrostu",
    r"utrzym\w* się wzrost\w* produkcj\w*",
    r"utrzym\w* się wzrost\w* sprzedaż\w*",
    r"inflacj\w* przyspieszył\w*",
    r"inflacj\w* zwiększył\w* się",
    r"dynamik\w* cen\w* wzrosł\w*",
    r"nasil\w* się presj\w* cenow\w*",
    r"rosnąc\w* presj\w* cenow\w*",
    r"wzrost\w* presj\w* cenow\w*",
    r"utrzym\w* się podwyższon\w* inflacj\w*",
    r"uporczyw\w* inflacj\w*",
    r"utrwaleni\w* oczekiwań inflacyjn\w*",
    r"odkotwiczeni\w* oczekiwań inflacyjn\w*",
    r"wzrost\w* inflacj\w* bazow\w*",
    r"przyspieszeni\w* inflacj\w* bazow\w*",
    r"wzrost\w* dynamik\w* wynagrodze\w*",
    r"przyspieszeni\w* wzrostu płac",
    r"nasileni\w* presj\w* płacow\w*",
    r"niedobor\w* pracownik\w*",
    r"napięci\w* na rynku pracy",
    r"spadk\w* stop\w* bezroboci\w*",
    r"zwiększeni\w* zatrudnieni\w*",
    r"ożywieni\w* gospodarcz\w*",
    r"ożywieni\w* koniunktur\w*",
    r"przyspieszeni\w* dynamik\w* PKB",
    r"siln\w* popyt\w* konsumpcyjn\w*",
    r"rosnąc\w* popyt\w* krajow\w*",
    r"wzrost\w* wykorzystani\w* mocy produkcyjn\w*",
    r"wzrost\w* cen\w* usług",
    r"presj\w* popytow\w*",
    r"ryzyko przegrzania gospodark\w*",
    r"wzrost\w* ryzyk\w* inflacyjn\w*",
    r"przewag\w* ryzyk\w* w górę",
    r"koniecznoś\w* zacieśnieni\w* polityk\w* pieniężn\w*",
    r"utrzymani\w* restrykcyjn\w* polityk\w* pieniężn\w*",
    r"ograniczeni\w* przestrzeni do obniż\w* stóp",
    r"osłabieni\w* tendencj\w* dezinflacyjn\w*",
    r"ograniczeni\w* tendencj\w* dezinflacyjn\w*",
    r"przyspieszeni\w* aktywności gospodarcz\w*",
    r"przyspieszając\w* wzrost\w* popyt\w*",
    r"stopniow\w* popraw\w* koniunktur\w*",
    r"kontynuacj\w* ożywieni\w* gospodarcz\w*",
    r"popraw\w* sytuacj\w* finansow\w* przedsiębiorstw",
    r"stabiln\w* wzrost\w* akcj\w* kredytow\w*",
    r"przyspieszeni\w* wzrostu popyt\w*",
    r"wyższ\w* dynamik\w* PKB",
    r"dobr\w* sytuacj\w* na rynku pracy",
    r"wzrost\w* inflacj\w* powyżej górnej granicy",
    r"inflacj\w* powyżej górnej granicy",
    r"powyżej górnej granicy odchyleń od celu",
    r"przekroczeni\w* górnej granicy odchyleń",
    r"wyższ\w* cen\w* paliw",
    r"wzrost\w* cen\w* energii elektrycznej",
    r"wzrost\w* cen\w* żywności",
    r"wzrost\w* cen\w* opłat",
    r"podwyższeni\w* akcyz\w*",
    r"wzrost\w* popyt\w* konsumpcyjn\w*",
    r"korzystn\w* sytuacj\w* finansow\w* gospodarstw domowych",
    r"dobr\w* nastroj\w* konsumentów",
    r"korzystn\w* sytuacj\w* na rynku pracy",
    r"korzystn\w* perspektyw\w* koniunktur\w*",
    r"dodatni\w* wkład\w* eksportu netto",
    r"utrzym\w* się inflacj\w* powyżej celu",
    r"wzrost\w* cen\w* surowc\w* żywnościow\w*",
    r"podwyższ\w* inflacj\w*",
    r"istotn\w* wzrost\w* inflacj\w*",
    r"wyraźn\w* wzrost\w* wynagrodze\w*",
    r"wzrosł\w* dynamik\w* sprzedaż\w*",
    r"wzrosł\w* dynamik\w* produkcj\w*",
    r"trwa\w* ożywieni\w* gospodarcz\w*",
    r"trwając\w* ożywieni\w* gospodarcz\w*",
    r"dalsz\w* ożywieni\w* aktywności gospodarcz\w*",
    r"korzystn\w* sytuacj\w* gospodarcz\w*",
    r"sytuacj\w* na rynku pracy poprawia się",
    r"wzrost\w* dochod\w* gospodarstw domowych",
    r"siln\w* wzrost\w* cen\w* transportu",
    r"wzrost\w* cen\w* gazu",
    r"wzrost\w* cen\w* surowc\w* energetyczn\w*",
    r"wyższ\w* cen\w* surowc\w*",
    r"znaczn\w* wyższ\w* cen\w*",
    r"podbij\w* dynamik\w* cen\w*",
    r"podwyższ\w* dynamik\w* cen\w*",
    r"inflacj\w* utrzym\w* się na podwyższon\w* poziomie",
    r"inflacj\w* może utrzymać się na podwyższon\w* poziomie",
    r"podwyższon\w* poziomie dłużej niż",
    r"dłużej niż dotychczas oczekiwano",
    r"ryzyko utrwaleni\w* się dynamik\w* cen\w*",
    r"utrwaleni\w* się dynamik\w* cen\w*",
    r"utrwaleni\w* inflacj\w* powyżej celu",
    r"poziom\w* wyższ\w* od celu inflacyjn\w*",
    r"ryzyko utrzymania się inflacj\w* powyżej celu",
    r"dodatnio na dynamik\w* cen\w* oddziałuj\w*",
    r"podwyższ\w* podstawow\w* stop\w* rezerw\w* obowiązkow\w*",
    r"podwyższ\w* stopy procentowe NBP",
    r"rośni\w* inflacj\w*",
    r"inflacj\w* rośni\w*",
    r"inflacj\w* osiąg\w* najwyższ\w* poziom\w*",
    r"najwyższ\w* poziom\w* inflacj\w*",
    r"wysok\w* dynamik\w* cen\w*",
    r"istotn\w* podwyższon\w* inflacj\w*",
    r"inflacj\w* pozost\w* podwyższon\w*",
    r"podwyższan\w* prognoz\w* inflacj\w*",
    r"podwyższeni\w* prognoz\w* inflacj\w*",
    r"wzrost\w* prognoz\w* inflacj\w*",
    r"rosnąc\w* koszt\w* pracy",
    r"wzrost\w* koszt\w* pracy",
    r"rośni\w* inflacj\w* bazow\w*",
    r"wysok\w* cen\w* surowc\w* energetyczn\w*",
    r"wzrost\w* cen\w* surowc\w* roln\w*",
    r"wzrost\w* taryf\w* na energi\w*",
    r"wzrost\w* cen\w* gazu ziemnego",
    r"wzrost\w* cen\w* energi\w* ciepln\w*",
    r"przedłużając\w* się zaburzeni\w* podaży",
    r"utrzymując\w* się zaburzeni\w* w łańcuchach dostaw",
    r"siln\w* wzrost\w* popyt\w*",
    r"wzrost\w* popyt\w* stymulowan\w* dochod\w*",
    r"wysok\w* dynamik\w* aktywności gospodarcz\w*",
    r"wysok\w* dynamik\w* sprzedaż\w*",
    r"wysok\w* dynamik\w* produkcj\w*",
    r"nadal rośni\w* zatrudnieni\w*",
    r"utrzym\w* się korzystn\w* koniunktur\w*",
    r"ryzyko kształtowani\w* się inflacj\w* powyżej celu",
    r"ryzyko utrwaleni\w* się podwyższon\w* inflacj\w*",
    r"ryzyko utrwaleni\w* podwyższon\w* inflacj\w*",
    r"utrzym\w* się ryzyko inflacj\w* powyżej celu",
    r"ograniczeni\w* ryzyka utrwaleni\w* inflacj\w*",
    r"ponown\w* podwyższ\w* stopy procentowe",
    r"kontynuacj\w* zacieśniani\w* polityk\w* pieniężn\w*",
    r"zaostrzeni\w* warunk\w* monetarn\w*",
    r"bardzo dobr\w* nastroj\w* konsumen\w*",
    r"zwiększając\w* się zatrudnieni\w*",
    r"rosnąc\w* konsumpcj\w*",
    r"ożywieni\w* nakład\w* inwestycyjn\w*",
    r"wyraźn\w* wzrost\w* inwestycj\w*",
    r"wzrosł\w* dynamik\w* cen\w* konsumpcyjn\w*",
    r"wyższ\w* inflacj\w* bazow\w*",
    r"zwiększył\w* się inflacj\w* bazow\w*",
    r"wyższ\w* dynamik\w* PKB niż oczekiwano",
    r"zdrożał\w* produkt\w* żywnościow\w*",
    r"wzrost\w* roczn\w* dynamik\w* cen\w* paliw",
    r"dynamik\w* wynagrodze\w*(?:\s+\S+){0,4}?\s+na wysok\w* poziomie",
    r"inflacj\w*(?:\s+\S+){0,4}?\s+powyżej celu",
    r"podwyższon\w* pozostan\w*(?:\s+\S+){0,2}?\s+inflacj\w* bazow\w*",
    r"przedłużeni\w* okresu pozostawani\w* inflacj\w* powyżej celu",
    r"wpływ\w* podwyższon\w* inflacj\w* na oczekiwani\w* inflacyjn\w*",
    r"wzrosł\w* (?:również\s+)?inflacj\w* bazow\w*",
    r"wzrost\w* roczn\w* wskaźnik\w* inflacj\w*",

]


dovish_words = [
    r"spadek\w*(?:\s+\S+){0,3}?\s+inflacj\w*",
    r"inflacj\w* spadł\w*",
    r"obniż\w* inflacj\w*",
    r"malejąc\w* inflacj\w*",
    r"spowolnieni\w* inflacj\w*",
    r"dezinflacj\w*",
    r"inflacj\w* poniżej celu",
    r"inflacj\w* niższ\w* od oczekiwa\w*",
    r"spadek\w* presj\w* inflacyjn\w*",
    r"słabnąc\w* presj\w* inflacyjn\w*",
    r"osłabieni\w* presj\w* inflacyjn\w*",
    r"ograniczeni\w* presj\w* inflacyjn\w*",
    r"spadek\w* oczekiwań inflacyjn\w*",
    r"obniż\w* oczekiwani\w* inflacyjn\w*",
    r"spadek\w*(?:\s+\S+){0,3}?\s+cen\w*",
    r"obniżył\w* się dynamik\w* wynagrodze\w*",
    r"spowolnieni\w* wzrostu wynagrodze\w*",
    r"spowolnieni\w* wzrostu płac",
    r"zatrudnieni\w* niższ\w* niż rok wcześniej",
    r"spadek\w* zatrudnieni\w* w sektorze przedsiębiorstw",
    r"obniżył\w* się dynamik\w* PKB",
    r"dynamik\w* PKB obniżył\w* się",
    r"produkcj\w* budowlan\w* obniżył\w* się",
    r"obniżeni\w* prognoz\w* inflacj\w*",
    r"rewizj\w* prognoz\w* inflacj\w* w dół",
    r"inflacj\w* obniży\w* się poniżej górnej granicy",
    r"obniżeni\w* inflacj\w* poniżej górnej granicy",
    r"spadek\w* inflacj\w* poniżej górnej granicy",
    r"powrót inflacj\w* do celu",
    r"trwał\w* powrót inflacj\w* do celu",
    r"ograniczeni\w* presj\w* popytow\w*",
    r"zmniejszeni\w* presj\w* popytow\w*",
    r"spadek\w* dynamik\w* cen\w* usług",
    r"niższ\w* prognoz\w* inflacj\w*",
    r"obniż\w* cen\w*",
    r"obniżeni\w* dynamik\w* cen\w*",
    r"spadek\w* cen\w* ropy\w*",
    r"obniż\w* cen\w* ropy\w*",
    r"spadek\w* cen\w* surowc\w*",
    r"spadek\w* wynagrodze\w*",
    r"spowolnieni\w* dynamik\w* płac\w*",
    r"obniżeni\w* dynamik\w* wynagrodze\w*",
    r"słabnąc\w* presj\w* płacow\w*",
    r"spadek\w*(?:\s+\S+){0,3}?\s+zatrudnieni\w*",
    r"pogorszeni\w* sytuacj\w* na rynku pracy",
    r"wzrost\w* bezroboci\w*",
    r"wzrost\w* stop\w* bezroboci\w*",
    r"spadek\w* popyt\w* na prac\w*",
    r"pogorszeni\w* perspektyw\w* zatrudnieni\w*",
    r"umiarkowan\w* podwyż\w* płac\w*",
    r"brak\w* efekt\w* drugiej rundy",
    r"spowolnieni\w* gospodarcz\w*",
    r"spadek\w* wzrostu gospodarcz\w*",
    r"osłabieni\w*(?:\s+\S+){0,2}?\s+koniunktur\w*",
    r"pogorszeni\w* koniunktur\w*",
    r"pogorszeni\w* nastroj\w*",
    r"spadek\w* produkcj\w*",
    r"spadek\w* sprzedaż\w*",
    r"obniżeni\w* dynamik\w* produkcj\w*",
    r"obniżeni\w* dynamik\w* sprzedaż\w*",
    r"spadek\w* inwestycj\w*",
    r"spadek\w* popyt\w* krajow\w*",
    r"spadek\w* konsumpcj\w*",
    r"spadek\w* zamówie\w*",
    r"obniżeni\w* wykorzystani\w* mocy wytwórcz\w*",
    r"pogorszeni\w* perspektyw\w* wzrostu",
    r"słab\w* aktywnoś\w* gospodarcz\w*",
    r"aprecjacj\w* złot\w*",
    r"relatywn\w* nisk\w* tempo wzrostu",
    r"tempo wzrostu\w* pozost\w* relatywn\w* nisk\w*",
    r"słab\w* nastroj\w* w sektorze przemysłow\w*",
    r"ryzyko osłabieni\w* koniunktur\w*",
    r"podwyższon\w* niepewnoś\w* dotycząc\w* koniunktur\w*",
    r"umiarkowan\w* poziom\w* inflacj\w*",
    r"inflacj\w* utrzym\w* się na umiarkowan\w* poziomie",
    r"inflacj\w* utrzym\w* się w pobliżu celu",
    r"inflacj\w* kształtow\w* się w pobliżu celu",
    r"osłabieni\w* koniunktur\w* w otoczeni\w* gospodarcz\w*",
    r"obniżył\w* się cen\w* ropy",
    r"umocnieni\w* złot\w*",
    r"spadek\w* cen\w* import\w*",
    r"spadek\w* cen\w* energi\w*",
    r"łagodn\w* nastawieni\w*",
    r"złagodzeni\w* nastawieni\w*",
    r"łagodzeni\w* polityk\w* pieniężn\w*",
    r"poluzowani\w* polityk\w* pieniężn\w*",
    r"obniż\w* stóp procentow\w*",
    r"spadek\w* stóp procentow\w*",
    r"przestrze\w* do obniż\w* stóp",
    r"ograniczeni\w* ryzyka inflacyjn\w*",
    r"ryzyko inflacji poniżej celu",
    r"cen\w* ropy\w* spadł\w*",
    r"cen\w* ropy\w* obniżył\w* się",
    r"spadku cen\w*",
    r"pogorszył\w* się perspektyw\w* wzrostu",
    r"utrzym\w* się spadek\w* produkcj\w*",
    r"inflacj\w* obniżył\w* się",
    r"inflacj\w* zmniejszył\w* się",
    r"inflacj\w* wyhamował\w*",
    r"dynamik\w* cen\w* spadł\w*",
    r"zmniejszeni\w* presj\w* cenow\w*",
    r"ustępowani\w* presj\w* inflacyjn\w*",
    r"wygasani\w* presj\w* inflacyjn\w*",
    r"ograniczeni\w* presj\w* cenow\w*",
    r"spadek\w* inflacj\w* bazow\w*",
    r"obniżeni\w* inflacj\w* bazow\w*",
    r"zakotwiczeni\w* oczekiwań inflacyjn\w*",
    r"stabilizacj\w* oczekiwań inflacyjn\w*",
    r"spadek\w* dynamik\w* wynagrodze\w*",
    r"wyhamowani\w* wzrostu płac",
    r"ograniczeni\w* presj\w* płacow\w*",
    r"osłabieni\w* popyt\w* na prac\w*",
    r"wzrost\w* liczby bezrobotn\w*",
    r"spadk\w* zatrudnieni\w*",
    r"spowolnieni\w* dynamik\w* PKB",
    r"osłabieni\w* wzrostu gospodarcz\w*",
    r"wyhamowani\w* aktywności gospodarcz\w*",
    r"spadek\w* popyt\w* konsumpcyjn\w*",
    r"słabnąc\w* popyt\w* krajow\w*",
    r"ograniczeni\w* inwestycj\w*",
    r"spadek\w* wykorzystani\w* mocy produkcyjn\w*",
    r"spadek\w* cen\w* usług",
    r"osłabieni\w* presj\w* popytow\w*",
    r"wzrost\w* ryzyka spowolnieni\w*",
    r"ryzyko recesj\w*",
    r"ryzyko inflacj\w* poniżej celu",
    r"przestrze\w* do łagodzeni\w* polityk\w* pieniężn\w*",
    r"możliwoś\w* obniżeni\w* stóp",
    r"ograniczeni\w* restrykcyjności polityk\w* pieniężn\w*",
    r"zmniejszeni\w* ryzyk\w* inflacyjn\w*",
    r"deflacj\w*",
    r"utrzym\w* się deflacj\w*",
    r"brak\w* presj\w* popytow\w*",
    r"nie ma presj\w* popytow\w*",
    r"nisk\w* dynamik\w* cen\w*",
    r"ujemn\w* dynamik\w* cen\w*",
    r"dynamik\w* cen\w* pozost\w* ujemn\w*",
    r"nisk\w* oczekiwani\w* inflacyjn\w*",
    r"bardzo nisk\w* oczekiwani\w* inflacyjn\w*",
    r"umiarkowan\w* dynamik\w* płac\w*",
    r"umiarkowan\w* tempo wzrostu popyt\w*",
    r"niski\w* cen\w* surowc\w*",
    r"nisk\w* cen\w* surowc\w*",
    r"siln\w* spadk\w* cen\w* surowc\w*",
    r"długotrwał\w* spadk\w* cen\w*",
    r"ograniczając\w* na aktywnoś\w* gospodarcz\w*",
    r"niekorzystn\w* perspektyw\w* gospodarcz\w*",
    r"niski\w* dynamik\w* aktywności",
    r"ryzyko utrzymania się inflacj\w* poniżej celu",
    r"dekoniunktur\w* w przemyśle",
    r"osłabion\w* aktywnoś\w* w przemyśle",
    r"osłabieni\w* aktywności gospodarcz\w*",
    r"dynamik\w* aktywności gospodarcz\w* osłab\w*",
    r"obniżani\w* się tempa wzrostu gospodarcz\w*",
    r"obniżył\w* się dynamik\w* wzrostu gospodarcz\w*",
    r"spowolnieni\w* tempa wzrostu gospodarcz\w*",
    r"wolniejsz\w* wzrost\w* konsumpcj\w*",
    r"wzrost\w* niepewności dotycząc\w* perspektyw",
    r"zwiększył\w* się niepewnoś\w*",
    r"pogorszył\w* się nastroj\w* na rynkach finansowych",
    r"osłabieni\w* koniunktur\w* za granicą",
    r"trwałości osłabieni\w* koniunktur\w*",
    r"niższ\w* dynamik\w* PKB",
    r"dynamik\w* PKB niższ\w*",
    r"wygaśnięci\w* wpływu czynnik\w*",
    r"wygaśnięci\w* czynnik\w* inflacyjn\w*",
    r"inflacj\w* będzie się obniżać",
    r"obniżani\w* się inflacj\w*",
    r"obniżeni\w* tempa wzrostu PKB",
    r"utrzymani\w* luźn\w* polityk\w* pieniężn\w*",
    r"skup\w* aktyw\w*",
    r"osłabieni\w* dynamik\w* aktywności",
    r"ograniczeni\w* podażow\w*",
    r"osłabieni\w* nastroj\w* w przemyśle",
    r"niższ\w* zatrudnieni\w* niż przed pandemią",
    r"wygaśni\w* czynnik\w* podażow\w*",
    r"wygaśni\w* oddziaływani\w* czynnik\w*",
    r"wygaśni\w* wpływ\w* czynnik\w*",
    r"ustępowani\w* zaburze\w* podażow\w*",
    r"normalizacj\w* łańcuch\w* dostaw",
    r"osłabieni\w* dynamik\w* sprzedaż\w*",
    r"osłabieni\w* dynamik\w* produkcj\w*",
    r"dynamik\w* wynagrodze\w*(?:\s+\S+){0,14}?\s+obniżył\w* się",
    r"inflacj\w* po wyłączeniu cen żywności i energii(?:\s+\S+){0,6}?\s+obniżył\w*",
    r"stopniow\w* obniżani\w* się dynamik\w* płac\w*",
    r"dynamik\w* cen\w* produkcj\w* sprzedan\w* przemysł\w*(?:\s+\S+){0,3}?\s+ujemn\w*",
    r"dynamik\w* aktywności gospodarcz\w*(?:\s+\S+){0,8}?\s+niższ\w* od oczekiwań",
    r"dynamik\w* (?:sprzedaż\w*|produkcj\w*)(?:\s+\S+){0,6}?\s+ujemn\w*",
    r"obniżani\w* się dynamik\w* płac\w*",
    r"inflacj\w*(?:\s+\S+){0,8}?\s+niższ\w* od oczekiwań",
    r"obniżył\w* się inflacj\w* po wyłączeniu cen energii i żywności",
    r"inflacj\w*(?:\s+\S+){0,6}?\s+niższ\w* niż wcześniej oczekiwano",
    r"presj\w* popytow\w* i kosztow\w*(?:\s+\S+){0,5}?\s+relatywn\w* nisk\w*",
    r"ograniczeni\w* krajow\w* presj\w* inflacyjn\w*",
    r"umocnieni\w* kursu złotego",
    r"dynamik\w* PKB(?:\s+\S+){0,4}?\s+obniżył\w*",
    r"dynamik\w* wynagrodze\w*(?:\s+\S+){0,8}?\s+niższ\w* niż kwartał wcześniej",
    r"dalsz\w* spadek\w* zatrudnieni\w*",
    r"ujemn\w*\s+roczn\w*\s+dynamik\w*\s+produkcj\w*\s+przemysłow\w*",
    r"ujemn\w*\s+roczn\w*\s+dynamik\w*\s+produkcj\w*\s+budowlano[-\s]?montażow\w*",
    r"zatrudnieni\w*(?:\s+\S+){0,12}?\s+niższ\w*\s+niż\s+rok\s+wcześniej",
    r"koniunktur\w*(?:\s+\S+){0,6}?\s+pozostaj\w*\s+osłabion\w*",
    r"inflacj\w*\s+powinn\w*\s+powróci\w*\s+do\s+celu",

]


# 1. SPRAWDZENIE REGEXÓW

for pattern in hawkish_words + dovish_words:
    re.compile(pattern)

print("All regex patterns compiled successfully!")


# 2. FUNKCJA ZLICZAJĄCA DOPASOWANIA

def count_matches(text, patterns):
    spans = []

    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            spans.append((match.start(), match.end()))

    spans.sort()

    if not spans:
        return 0

    count = 1
    last_end = spans[0][1]

    for start, end in spans[1:]:
        if start >= last_end:
            count += 1
            last_end = end
        else:
            last_end = max(last_end, end)

    return count


# 3. LICZBA SYGNAŁÓW HAWKISH I DOVISH

df["hawkish_count"] = df["clean_text"].fillna("").apply(
    lambda text: count_matches(text, hawkish_words)
)

df["dovish_count"] = df["clean_text"].fillna("").apply(
    lambda text: count_matches(text, dovish_words)
)


# 4. LICZBA SŁÓW W KOMUNIKATACH

df["word_count"] = (
    df["clean_text"]
    .fillna("")
    .str.split()
    .str.len()
)


# 5. INDEKS JASTRZĘBIOŚCI - RATIO

df["hawkish_index"] = (
    (df["hawkish_count"] - df["dovish_count"])
    / (df["hawkish_count"] + df["dovish_count"]).replace(0, np.nan)
)


# 6. INDEKS JASTRZĘBIOŚCI - DENSITY

df["hawkish_density"] = (
    1000
    * (df["hawkish_count"] - df["dovish_count"])
    / df["word_count"].replace(0, np.nan)
)


# 7. SORTOWANIE I WYŚWIETLENIE WYNIKÓW

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

print(
    df[
        [
            "date",
            "word_count",
            "hawkish_count",
            "dovish_count",
            "hawkish_index",
            "hawkish_density"
        ]
    ].tail(20).to_string(index=False)
)


# 8. WYKRES - HAWKISH INDEX (RATIO)

plt.figure(figsize=(14, 6))

plt.plot(
    df["date"],
    df["hawkish_index"],
    linewidth=1.5,
    label="Hawkishness Index - Ratio"
)

plt.axhline(0, color="gray", linestyle="--", linewidth=1)

plt.title("NBP Hawkishness Index - Ratio (2005-2026)")
plt.xlabel("Date")
plt.ylabel("Hawkishness Index")

plt.ylim(-1.05, 1.05)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# 9. WYKRES - HAWKISH INDEX (DENSITY)

plt.figure(figsize=(14, 6))

plt.plot(
    df["date"],
    df["hawkish_density"],
    linewidth=1.5,
    label="Hawkishness Index - Density"
)

plt.axhline(0, color="gray", linestyle="--", linewidth=1)

plt.title("NBP Hawkishness Index - Density (2005-2026)")
plt.xlabel("Date")
plt.ylabel("Net Hawkish Signals per 1000 Words")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
