Tilgå serveren lokalt. 

1. Kør main.py med "uvicorn main:app --reload --host 0.0.0.0 --port 8000"
2. Åben powershell og skriv ipconfig
3. Kik efter IPv4 Address noget i stil med "IPv4 Address. . . . . . : 192.168.1.xxx"
4. Skriv i en browser på din anden computer, som er på dit lokale netværk eller tilsluttet med VPN. "http://INDSÆT IPv4 Address:8000"
5. Du er nu på serveren skriv eventuelt /docs bagefter det, som står i browseren for at tilgå den midlertidige client. 



Lav requirements fil af conda miljø, ved at skrive pip freeze > requirements.txt