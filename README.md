# RAG System - PDF Multi-Upload

ß
### Tre sätt att arbeta med PDFs:

1. ** En PDF** - Ladda upp en enskild PDF (som tidigare, men förbättrad)
2. ** Flera PDFs** - Ladda upp många PDFs samtidigt till samma samling
3. ** Lägg till i befintlig** - Utöka en befintlig samling med nya PDFs

### Samma funktionalitet som för URLs!
Nu har både PDFs och URLs exakt samma arbetsflöde och möjligheter.

## Vad ingår?

### Uppdaterade filer:
- **server.py** - Backend med stöd för PDF multi-upload
- **index.html** - Frontend med tre PDF-lägen
- **helpers.py** - Oförändrad (inkluderad för komplettering)
- **rag_pipeline.py** - Oförändrad (inkluderad för komplettering)

### Dokumentation:
- **UPPDATERINGAR.md** - Detaljerad funktionsbeskrivning
- **INSTALLATION.md** - Installations- och uppgraderingsguide
- **README.md** - Denna fil

## Snabbstart

### Första användningen:
1. Öppna webbgränssnittet
2. Gå till PDF-sektionen
3. Se de tre nya lägena
4. Testa att ladda upp flera PDFs samtidigt!

## UI-förändringar

### Före:
```
PDF med Semantisk Sökning
├── Samlingens namn (valfritt)
├── Embedding Model
├── Chunk-storlek
└── Drag-and-drop zon
```

### Efter:
```
PDF med Semantisk Sökning
├── Mode Selector [En PDF | Flera PDFs | Lägg till i befintlig]
│
├── EN PDF
│   ├── Samlingens namn (valfritt)
│   ├── Embedding Model
│   ├── Chunk-storlek
│   └── Drag-and-drop zon
│
├── FLERA PDFs
│   ├── Samlingens namn (obligatoriskt)
│   ├── Embedding Model
│   ├── Chunk-storlek
│   ├── Drag-and-drop zon (multi-select)
│   └── Progress bar
│
└── LÄGG TILL I BEFINTLIG
    ├── Välj samling (dropdown)
    ├── Embedding Model
    ├── Chunk-storlek
    └── Drag-and-drop zon
```

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

## Tekniska Detaljer

### Backend-ändringar (server.py):
- `/api/upload_pdf` - Stödjer nu befintliga samlingar
- `/api/collections` - Returnerar `pdf_count`
- `/api/collection_info` - Returnerar `indexed_pdfs`
- Metadata tracking för PDFs (`indexed_pdfs`)

### Frontend-ändringar (index.html):
- Tre PDF-lägen med mode selector
- Multi-file upload support
- Progress bar för batch uploads
- PDF-lista i resultatvisning
- Drag-and-drop för alla lägen

## Kompatibilitet

- **Bakåtkompatibel** - Gamla samlingar fungerar utan ändringar
- **Befintliga funktioner** - Allt som fungerade före fungerar fortfarande
- **API-kompatibilitet** - Inga breaking changes
- **Metadata** - Automatisk uppdatering av befintliga samlingar

## Metadata-format

### Ny metadata-struktur:
```json
{
  "indexed_urls": [
    "https://example.com/page1",
    "https://example.com/page2"
  ],
  "indexed_pdfs": [
    "dokument1.pdf",
    "dokument2.pdf",
    "rapport.pdf"
  ],
  "total_records": 456,
  "total_vectors": 456,
  "last_updated": "2025-11-26T10:30:00Z",
  "embed_backend": "google"
}
```

## Kända begränsningar

- **Filstorlek**: Begränsas av server-konfiguration (default ~100MB)
- **Parallell processing**: Batch-uploads processas sekventiellt
- **Browser memory**: Många stora PDFs kan påverka prestanda

## Framtida förbättringar

Potentiella förbättringar (ej implementerade än):
- [ ] Parallell batch processing
- [ ] PDF preview i UI
- [ ] Ta bort enskilda PDFs från samling
- [ ] PDF metadata-extraction
- [ ] Merge collections
- [ ] Export/Import funktionalitet

## Dokumentation

- **[UPPDATERINGAR.md](UPPDATERINGAR.md)** - Detaljerad funktionsbeskrivning med exempel
- **[INSTALLATION.md](INSTALLATION.md)** - Steg-för-steg installation och felsökning

## Support

### Frågor?
Kontakta systemadministratören eller öppna ett issue.

### Buggrapporter
Inkludera:
- Browser och version
- Felmeddelande (från console/network tab)
- Steg för att återskapa felet
- Server-loggar

### Feature requests
Förslag på förbättringar är alltid välkomna!

## Changelog

### Version 2.0 (2025-11-26)
- Lagt till multi-PDF upload
- Lagt till "lägg till i befintlig" för PDFs
- Progress bar för batch uploads
- PDF-lista i resultatvisning
- Förbättrad error handling
- Komplett dokumentation

### Version 1.0 (Tidigare)
- Initial release med single PDF upload
- URL multi-upload funktionalitet
- Semantisk sökning
- AI-frågor

---
**Version**: 2.0  
**Senast uppdaterad**: 2025-11-26  
**Licens**: Public Domain
