# RAG Search System 

När man lägger till content/context gör man det till en s.k. samling. Till en samling kan man lägga innehåll från PDF-filer eller webbsidor från internet. En samling kan innehålla text från både PDF-filer och länkar.

När man väl har en samling kan man välja den för sökning bland all den information som samlingen innehåller.

### Två sätt att arbeta med PDFs:

Alla PDF:er kan laddas upp med dra-o-släpp

1. **Ny samling** - Ladda upp en eller mmånga PDFs samtidigt till en ny samling
2. **Utöka befintlig samling** - Utöka en befintlig samling med en eller flera nya PDFs

### Två sätt att arbeta med URLar:

URL:ar hanterar ännu inte dra-o-släpp. **OBS**, endast en URL per rad

1. **Ny samling** – Klistra in eller flera URL:ar till en ny samling
2. **Utöka befintlig samling** – Utöka en befintlig samling med en eller flera nya URL:ar


## Snabbstart

### Första användningen:

Se till att rätt python miljö finns tillgänglig. Till exempel med följande kommando i en terminal (/slutarbete). Precis vilka Python-bibliotek som behöver installeras är beskrivet i ett annat dokument.

        source .venv/bin/activate

Navigera till /app och starta den lokala servern med kommandot:

        uvicorn server:app --reload --host 0.0.0.0 --port 8000

Accessa tjänsten via:

        http://localhost:8000/


## Användningsexempel

### Exempel 1: Årsredovisningar
```
1. Välj "Flera PDFs"
2. Ange namn: "Årsredovisningar_2020-2024"
3. Dra och släpp alla årsredovisningar samtidigt
4. Alla indexeras till samma samling
```

### Exempel 2: Utöka dokumentation
```
1. Välj "Lägg till i befintlig"
2. Välj samling: "Projektdokumentation"
3. Ladda upp ny manual
4. Manualen läggs till i befintlig samling
```

### Exempel 3: Snabb single-upload
```
1. Välj "En PDF" (standard)
2. Dra och släpp PDF
3. Automatiskt samlingens namn från filnamn
```




---
**Version**: 0.1  
**Senast uppdaterad**: 2025-12-01  
**Licens**: Public Domain
