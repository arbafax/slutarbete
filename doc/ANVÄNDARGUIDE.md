**VIKTIGT** Denna dokumentation är automatgenererad och _**inte**_ verifierad 2025-12-01 15:01

# RAG Search System - Användarguide

En komplett guide för att använda RAG Search System efter installation.

## Innehållsförteckning

- [Översikt](#översikt)
- [Kom igång](#kom-igång)
- [Arbeta med PDF-filer](#arbeta-med-pdf-filer)
  - [Skapa ny samling från PDF](#skapa-ny-samling-från-pdf)
  - [Lägga till PDFs i befintlig samling](#lägga-till-pdfs-i-befintlig-samling)
  - [Välj embedding-modell för PDF](#välj-embedding-modell-för-pdf)
- [Arbeta med webbsidor](#arbeta-med-webbsidor)
  - [Extrahera från enstaka URL](#extrahera-från-enstaka-url)
  - [Extrahera från flera URLs](#extrahera-från-flera-urls)
  - [Lägga till URLs i befintlig samling](#lägga-till-urls-i-befintlig-samling)
- [Söka i samlingar](#söka-i-samlingar)
  - [Semantisk sökning](#semantisk-sökning)
  - [Förstå sökresultat](#förstå-sökresultat)
  - [Optimera sökningar](#optimera-sökningar)
- [Fråga AI](#fråga-ai)
  - [Ställa frågor](#ställa-frågor)
  - [Välja AI-modell](#välja-ai-modell)
  - [Tolka AI-svar](#tolka-ai-svar)
  - [Anpassa system-prompt](#anpassa-system-prompt)
- [Hantera samlingar](#hantera-samlingar)
  - [Visa samlingsstatistik](#visa-samlingsstatistik)
  - [Ta bort samlingar](#ta-bort-samlingar)
  - [Exportera samlingar](#exportera-samlingar)
- [Tips och bästa praxis](#tips-och-bästa-praxis)
- [Vanliga användningsfall](#vanliga-användningsfall)

---

## Översikt

RAG Search System är ett kraftfullt verktyg som kombinerar:
- **Dokumentbearbetning:** PDF-extrahering och webbscraping
- **AI-embeddings:** Intelligent vektorrepresentation av text
- **Semantisk sökning:** Hitta relevant information baserat på betydelse, inte bara nyckelord
- **AI-assisterat svar:** Få exakta svar på dina frågor från dina dokument

### Huvudfunktioner

1. **Extrahera från PDF:** Ladda upp PDFs och gör dem sökbara
2. **Extrahera från URL:** Scrapa webbsidor och lagra innehållet
3. **Sök i samling:** Semantisk sökning i dina dokument
4. **Fråga AI om samling:** Ställ frågor och få AI-genererade svar

---

## Kom igång

### Starta systemet

1. Öppna en terminal/kommandoprompt
2. Navigera till projektmappen
3. Aktivera virtuell miljö (om du använder en)
4. Starta servern:
   ```bash
   # Windows
   python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000

   # macOS/Linux
   python3 -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
   ```
5. Öppna webbläsaren och gå till: `http://localhost:8000`

### Förstå gränssnittet

Gränssnittet är uppdelat i fyra huvudsektioner:

1. **Extrahera från PDF** - Längst upp, för PDF-uppladdning
2. **Extrahera från URL** - För webbscraping
3. **Sök i samling** - För att söka i dina samlingar
4. **Fråga AI om samling** - För att ställa frågor till AI

---

## Arbeta med PDF-filer

### Skapa ny samling från PDF

1. **Välj läge:**
   - Klicka på "Ny samling" (ska vara aktivt som standard)

2. **Namnge samlingen:**
   - Ange ett beskrivande namn i "Samlingens namn"
   - Exempel: "Årsredovisning 2024", "Tekniska manualer", "Forskningsartiklar"
   - Om du lämnar tomt skapas namnet automatiskt från första filen

3. **Välj embedding-modell:**
   - **Google Gemini (Standard):** Bra allround-val, kräver Google API-nyckel
   - **Cohere v3 ⭐:** Robust mot brusig data, utmärkt för PDF med formatfel
   - **BGE-M3 ⭐:** State-of-the-art öppen källkod, mycket bra kvalitet
   - **E5 ⭐:** Multilingual, särskilt bra för svenska texter
   - **OpenAI:** Hög kvalitet, kräver OpenAI API-nyckel
   - **Sentence-BERT:** Enkel lokal lösning, ingen API-nyckel krävs
   - **Ollama:** Lokal AI, kräver Ollama-installation

4. **Välj chunk-storlek:**
   - **256 tokens:** Snabbare, använd för korta dokument eller snabb bearbetning
   - **512 tokens (Standard):** Balans mellan kontext och prestanda
   - **1024 tokens:** Mer kontext, använd för komplexa dokument

5. **Aktivera/avaktivera överlappande chunks:**
   - **Aktiverat (Standard):** Bättre kontextbevarande, rekommenderat
   - **Avaktiverat:** Snabbare bearbetning, mindre redundans

6. **Ladda upp filer:**
   - Klicka på filväljaren eller dra-och-släpp PDF-filer
   - Du kan ladda upp flera filer samtidigt
   - Filerna syns i en lista under filväljaren

7. **Starta bearbetning:**
   - Klicka "Ladda upp & Extrahera"
   - En förloppsindikator visar processen
   - Vänta tills alla filer är bearbetade

8. **Resultat:**
   - Efter bearbetning visas statistik:
     - Antal filer
     - Totalt antal chunks
     - Samlingens namn
   - Länk för att ladda ner JSON-data

### Lägga till PDFs i befintlig samling

1. **Byt läge:**
   - Klicka på "Lägg till i befintlig"

2. **Välj samling:**
   - Välj en befintlig samling från dropdown-menyn
   - Klicka "↻" för att uppdatera listan

3. **Välj inställningar:**
   - Chunk-storlek och överlappning kan anpassas
   - Embedding-modellen måste matcha den ursprungliga samlingen

4. **Ladda upp filer:**
   - Välj PDF-filer att lägga till
   - Klicka "Ladda upp & Extrahera"

5. **Resultat:**
   - Nya chunks läggs till i befintlig samling
   - Statistiken uppdateras

### Välj embedding-modell för PDF

#### När ska jag använda vilken modell?

**Google Gemini (Standard):**
- ✅ Bra för allmänt bruk
- ✅ Stöder många språk inkl. svenska
- ✅ Snabb och pålitlig
- ❌ Kräver internetanslutning
- ❌ Kostar om man överskrider gratiskvoter

**Cohere v3 ⭐ (Rekommenderad för PDFs):**
- ✅ Mycket robust mot brusig data
- ✅ Utmärkt för PDF med formatfel
- ✅ Stöder stora kontextlängder (512 tokens)
- ✅ Flerspråkig
- ❌ Kräver Cohere API-nyckel

**BGE-M3 ⭐ (Bästa kvalitet):**
- ✅ State-of-the-art prestanda
- ✅ Öppen källkod
- ✅ Utmärkt för akademiska texter
- ⚠️ Kräver mer beräkningskraft
- ❌ Långsammare än Google/Cohere

**E5 ⭐ (Bäst för svenska):**
- ✅ Optimerad för multilinguala texter
- ✅ Särskilt bra för svenska dokument
- ✅ Bra balans mellan kvalitet och hastighet
- ⚠️ Kräver lokal modellnedladdning

**Sentence-BERT (Enkel):**
- ✅ Ingen API-nyckel krävs
- ✅ Fungerar offline
- ✅ Snabb
- ❌ Lägre kvalitet än andra alternativ
- ❌ Mindre effektiv för svenska

**Ollama (Lokal):**
- ✅ Helt lokal, ingen data lämnar datorn
- ✅ Ingen API-kostnad
- ✅ Bra integritet
- ❌ Kräver Ollama-installation
- ❌ Långsammare

---

## Arbeta med webbsidor

### Extrahera från enstaka URL

1. **Välj läge:**
   - Klicka på "Ny samling" under "Extrahera från URL"

2. **Ange URL:**
   - Klistra in webbadressen (t.ex. `https://sv.wikipedia.org/wiki/Artificiell_intelligens`)
   - URL:en måste vara giltig och börja med `http://` eller `https://`

3. **Namnge samlingen:**
   - Ange ett namn för samlingen
   - Om du lämnar tomt används URL:ens titel som namn

4. **Välj inställningar:**
   - Embedding-modell (samma alternativ som för PDF)
   - Chunk-storlek
   - Överlappande chunks

5. **Starta extrahering:**
   - Klicka "Extrahera"
   - Systemet hämtar och bearbetar sidan
   - Visas förlopp under bearbetningen

6. **Resultat:**
   - Statistik visas när processen är klar
   - Antal chunks och information om innehållet

### Extrahera från flera URLs

1. **Välj läge:**
   - Klicka på "Ny samling"

2. **Lägg till flera URLs:**
   - Skriv in första URL:en
   - Klicka "+ Lägg till URL"
   - URL:en läggs till i listan nedanför
   - Upprepa för varje URL du vill lägga till

3. **Hantera URL-listan:**
   - Klicka på "×" för att ta bort en URL
   - Alla URLs i listan kommer att bearbetas

4. **Namnge samlingen:**
   - Ge ett beskrivande namn som omfattar alla URLs
   - Exempel: "AI-artiklar", "Företagsinfo", "Produktdokumentation"

5. **Starta extrahering:**
   - Klicka "Extrahera"
   - Alla URLs bearbetas sekventiellt
   - Förloppsindikator visar status

6. **Resultat:**
   - Statistik för hela samlingen
   - Alla URLs indexeras i samma samling

### Lägga till URLs i befintlig samling

1. **Byt läge:**
   - Klicka på "Lägg till i befintlig"

2. **Välj samling:**
   - Välj befintlig samling från dropdown
   - Klicka "↻" för att uppdatera listan

3. **Lägg till URLs:**
   - Ange enstaka URL eller
   - Lägg till flera URLs med "+ Lägg till URL"

4. **Starta extrahering:**
   - Klicka "Extrahera"
   - Nya URLs läggs till i befintlig samling

**Tips för URL-extrahering:**
- Kontrollera att webbsidan är publik och tillgänglig
- Vissa webbplatser kan blockera scraping
- Undvik sidor som kräver inloggning
- Välj sidor med strukturerat innehåll för bäst resultat

---

## Söka i samlingar

### Semantisk sökning

Semantisk sökning hittar resultat baserat på betydelse och kontext, inte bara exakta nyckelord.

1. **Välj samling:**
   - Öppna dropdown-menyn under "Sök i samling"
   - Välj den samling du vill söka i
   - Klicka "↻" för att uppdatera listan om din samling inte syns

2. **Skriv sökfråga:**
   - Skriv din fråga i naturligt språk
   - Exempel:
     - "Information om företagets finansiella resultat"
     - "Hur man installerar produkten"
     - "Vad säger dokumentet om hållbarhet?"

3. **Anpassa sökningen:**
   - **Antal resultat (k):** Välj hur många resultat du vill se (1-20)
   - **Embedding-modell:** Välj samma modell som användes när samlingen skapades

4. **Sök:**
   - Klicka "Sök" eller tryck Enter
   - Resultat visas omedelbart

### Förstå sökresultat

Varje sökresultat innehåller:

- **Rubrik:** Heading från dokumentet
- **Score:** Relevanspoäng (0-1, högre = mer relevant)
- **Text:** Relevant textavsnitt från dokumentet
- **Bakgrund:** Grönt för att indikera relevans

**Relevanspoäng:**
- **0.8-1.0:** Mycket relevant, exakt matchning
- **0.6-0.8:** Relevant, bra matchning
- **0.4-0.6:** Delvis relevant
- **0.0-0.4:** Låg relevans, överväg att omformulera frågan

### Optimera sökningar

**För bättre resultat:**

1. **Var specifik:**
   - Dåligt: "information"
   - Bra: "information om leveranstider för produkter"

2. **Använd kontext:**
   - Dåligt: "pris"
   - Bra: "vilket pris har premium-planen?"

3. **Ställ frågor:**
   - Dåligt: "installation"
   - Bra: "hur installerar jag programvaran på Windows?"

4. **Justera antal resultat:**
   - Få resultat (k=3): För specifika frågor
   - Många resultat (k=10-15): För bred översikt

5. **Prova omformulera:**
   - Om resultaten inte är bra, prova olika formuleringar
   - Använd synonymer eller relaterade begrepp

---

## Fråga AI

### Ställa frågor

AI-funktionen använder sökresultaten för att generera exakta, kontextbaserade svar.

1. **Välj samling:**
   - Välj samling från dropdown under "Fråga AI om samling"

2. **Formulera fråga:**
   - Skriv din fråga i naturligt språk
   - Exempel:
     - "Vad är huvudpunkterna i rapporten?"
     - "Sammanfatta företagets finansiella ställning"
     - "Vilka krav ställs för att få bidrag?"

3. **Anpassa parametrar:**
   - **Antal källor (k):** Hur många dokument AI ska analysera (3-10)
   - **LLM-modell:** Välj AI-modell att använda

4. **Ställ fråga:**
   - Klicka "Fråga AI"
   - En laddningsindikator visas medan AI tänker
   - Svar genereras baserat på relevanta dokument

### Välja AI-modell

**Google Gemini 2.0 Flash (Standard):**
- ✅ Snabb och effektiv
- ✅ Bra för svenska
- ✅ Hög kvalitet
- ✅ Del av gratis Google AI Studio
- Rekommenderas för de flesta användningsfall

**OpenAI GPT-4o-mini:**
- ✅ Mycket hög kvalitet
- ✅ Utmärkt resonemangsförmåga
- ✅ Bra för komplexa frågor
- ❌ Kräver OpenAI API-nyckel och kostar mer

**Ollama Llama3.2 (Lokal):**
- ✅ Helt lokalt, ingen data skickas ut
- ✅ Ingen API-kostnad
- ✅ Bra integritet
- ❌ Kräver Ollama-installation
- ❌ Långsammare än molntjänster
- ⚠️ Lägre kvalitet än Google/OpenAI

### Tolka AI-svar

**Svar-sektionen innehåller:**

1. **AI-genererat svar:**
   - Skrivet i naturlig prosa
   - Baserat endast på dokument i samlingen
   - Citera inte information utanför samlingen

2. **Källor:**
   - Visar vilka dokument AI baserade svaret på
   - Varje källa inkluderar:
     - Rubrik/heading
     - Relevanspoäng
     - Textförhandsvisning
   - Högre poäng = mer central för svaret

**Om AI säger "inte tillräcklig information":**
- Dokumentet innehåller inte svar på frågan
- Prova omformulera frågan
- Lägg till fler dokument till samlingen
- Öka antal källor (k-värdet)

### Anpassa system-prompt

System-promten styr hur AI svarar. Standard-promten instruerar AI att:
- Svara endast baserat på given kontext
- Inte gissa eller hitta på information
- Vara tydlig och koncis
- Dela upp svaret i läsbara stycken

**För att anpassa:**

1. I webbgränssnittet finns ett fält för "Anpassad system-prompt"
2. Skriv egna instruktioner, t.ex.:
   ```
   Du är en teknisk expert som svarar på frågor om produktdokumentation.
   Ge alltid konkreta exempel och steg-för-steg instruktioner.
   Om information saknas, föreslå vad användaren kan göra istället.
   ```

**Tips för system-prompt:**
- Var specifik om ton och stil
- Ge exempel på önskat format
- Specificera vad AI ska göra om information saknas
- Håll det koncist (under 200 ord)

---

## Hantera samlingar

### Visa samlingsstatistik

Under "Sök i samling" finns "Samlingsöversikt":

- **Namn:** Samlingens namn
- **Laddad:** Om samlingen är laddad i minnet
- **URLs:** Antal indexerade webbsidor
- **PDFs:** Antal indexerade PDF-filer
- **Totalt chunks:** Antal textsegment i samlingen

Klicka "↻ Uppdatera samlingar" för att hämta senaste status.

### Ta bort samlingar

1. I "Samlingsöversikt", klicka på "🗑" (papperskorgen) bredvid samlingen
2. Bekräfta radering
3. Samlingen och alla dess data raderas permanent

**Varning:** Detta kan inte ångras! Säkerhetskopiera viktiga samlingar först.

### Exportera samlingar

Efter bearbetning av PDF eller URL visas en nedladdningslänk:

1. Klicka på länken "Ladda ner som JSON"
2. Filen laddas ner i JSON-format
3. Innehåller all extraherad text och metadata

**JSON-filen innehåller:**
- Källinformation (URL eller filnamn)
- Titel
- Alla chunks med text och metadata
- Embeddings (vektorer)
- Strukturinformation (headings, nivåer)

---

## Tips och bästa praxis

### Allmänna tips

1. **Namngivning:**
   - Använd beskrivande namn för samlingar
   - Inkludera datum för versionerade dokument
   - Exempel: "Årsredovisning_2024", "Produktmanualer_Q4"

2. **Organisering:**
   - Håll relaterade dokument i samma samling
   - Skapa separata samlingar för olika ämnesområden
   - Undvik att blanda orelaterat innehåll

3. **Chunk-storlek:**
   - Mindre chunks (256): Snabbare, bättre för korta frågor
   - Större chunks (512-1024): Mer kontext, bättre för komplexa dokument

4. **Överlappande chunks:**
   - Aktivera för bättre kontextbevarande
   - Särskilt viktigt för långa dokument
   - Avaktivera endast om du behöver maximal hastighet

### För PDF-bearbetning

1. **Förbered PDFs:**
   - Undvik skannade PDFs med dålig OCR-kvalitet
   - Välj text-baserade PDFs när möjligt
   - Ta bort ovidkommande sidor (t.ex. omslag, tomma sidor)

2. **Val av embedding-modell:**
   - Cohere för PDFs med formatfel
   - BGE-M3 för akademiska artiklar
   - E5 för svenska dokument

3. **Batchbearbetning:**
   - Ladda upp flera relaterade PDFs samtidigt
   - Systemet bearbetar dem sekventiellt
   - Kontrollera förloppsindikator

### För URL-extrahering

1. **Välj rätt sidor:**
   - Artikelsidor fungerar bäst
   - Undvik sidor med mycket JavaScript
   - Statiska sidor ger bättre resultat än dynamiska

2. **Flera URLs:**
   - Gruppera relaterade sidor i samma samling
   - Lägg till dokumentationssidor systematiskt
   - Använd genomtänkt namngivning

3. **Uppdatera innehåll:**
   - Skapa ny samling för uppdaterat innehåll
   - Eller lägg till nya versioner i befintlig samling
   - Radera gamla samlingar när de inte längre är relevanta

### För sökning och frågor

1. **Iterativ process:**
   - Börja med bred fråga
   - Förfina baserat på resultat
   - Justera k-värdet efter behov

2. **Kombinera funktioner:**
   - Använd semantisk sökning först för att hitta relevant innehåll
   - Ställ sedan specifika frågor till AI baserat på vad du hittat

3. **Kvalitetskontroll:**
   - Kontrollera alltid källor
   - Verifiera AI-svar mot originaldokument
   - Använd relevanspoäng som indikator

---

## Vanliga användningsfall

### Forskare / Akademiker

**Scenario:** Analysera forskningsartiklar

1. Ladda upp PDF-artiklar till en samling "Forskning_MLsystem"
2. Använd BGE-M3 för högsta kvalitet
3. Sök efter specifika koncept: "transformer-arkitektur"
4. Fråga AI: "Sammanfatta de senaste framstegen inom transformers"

### Företagsanvändare

**Scenario:** Analysera årsredovisningar

1. Ladda upp årsredovisningar från flera år
2. Namnge: "Årsredovisningar_2020-2024"
3. Sök: "försäljningsutveckling"
4. Fråga AI: "Hur har lönsamheten utvecklats de senaste fem åren?"

### Studenter

**Scenario:** Sammanfatta kursmaterial

1. Samla PDFs från kursens föreläsningar
2. Skapa samling: "Kursmaterial_AI_Grundkurs"
3. Sök efter specifika begrepp inför tenta
4. Fråga AI: "Förklara skillnaden mellan supervised och unsupervised learning"

### Produktteam

**Scenario:** Dokumentationsportal

1. Scrapa alla dokumentationssidor från företagets wiki
2. Skapa samling: "Produktdokumentation_v2"
3. Gör sökning tillgänglig för support-teamet
4. Låt AI besvara vanliga frågor automatiskt

### Legal / Compliance

**Scenario:** Policydokument

1. Ladda upp alla policy- och compliance-dokument
2. Använd Cohere (robust mot formatfel)
3. Sök efter specifika regler
4. Fråga AI: "Vilka krav ställs för GDPR-compliance?"

---

## Nästa steg

- För tekniska detaljer, se [TEKNISK_DOKUMENTATION.md](TEKNISK_DOKUMENTATION.md)
- För installationshjälp, se [INSTALLATIONSGUIDE.md](INSTALLATIONSGUIDE.md)
- För felsökning, se "Vanliga problem" i installationsguiden

---

**Lycka till med RAG Search System!** 🎯
