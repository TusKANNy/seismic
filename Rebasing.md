## Questo documento tiene traccia del cherry pick di main dentro ecir2026/streamvbyte. 
C'erano circa 40 commit di cui fare cherry pick. Molti erano di documentazioni e piccoli bug fix, e sono stati integrati correttamente. Alcuni portavano a un conflitto strutturale, quindi sono stati ignorati. In particolare, in `main`, l'InvertedIndex era parametrizzato da `ComponentType` e `ValueType`. Invece, nel branch piu' nuovo, e' parametrizzato da uno Sparse Dataset. 

Per ogni commit, ho annotato se il merge e' stato fatto oppure no ed eventualmente quali modifiche sono state fatte.

- 3256087 2025-05-14 Update Guidelines.md
    - [Done, Manual Merge]

- b9d8ebb 2025-05-14 Update Guidelines.md
    - [Done, Auto merge]

- ea1f31e 2025-05-14 Update Guidelines.md
    - [Done, Auto merge]

- 9281361 2025-05-14 Update Guidelines.md
    - [Done, Auto merge]

- 5bc746c 2025-05-15 Update RunExperiments.md
    - [Skip, modifiche già presenti]

- 2f1327a 2025-06-06 Update README.md
    - [Skip, modifiche già presenti]

- aaa552 2025-06-11 Merge branch 'main' of https://github.com/TusKANNy/seismic
    - [Ignored]

- 04574e4 2025-06-18 Merge branch 'develop'
    - [Ignored]

- f0a55d1 2025-07-22 Merge remote-tracking branch 'public/develop'
    - [Ignored]

- 9015083 2025-01-05 Various improvements
    - [Drammatico, Ignorato]
    - Questo commit introdurrebbe cambiamenti distruttivi.
    - Nel merge, il branch main cerca di riportare alla vecchia struttura dove l'Inverted Index è parametrizzato da ComponentType e ValueType,
      anziché da uno SparseDatasetTrait.
    - Questa è la struttura attuale anche nel main pubblico.
    - Da controllare le distanze, ma quelle implementate in questo branch dovrebbero comunque essere molto veloci.
    - Verrebbero comunque rimpiazzate da Vectorium.

- 9cf1069 2025-09-09 merging Martino's improvements. Wow!
    - [Drammatico, Ignorato]
    - Questo commit lavora a monte del precedente e fa un po' di pulizia.
    - Rimuove qwt.
    - Dettagli:
        - RIMUOVE la dipendenza qwt da Cargo.toml
        - Bump versione (0.2.3 → 0.4.0)
        - Fix in utils.rs: query_components.len() < 2^18 → query_dim <= 2^18
        - Cambiamenti cosmetici in inverted_index.rs

- 2e2e0bd 2025-08-13 HACK: Faster Fixed point numbers to f32 conversion
    - [Ignorato]
    - Controllare se può essere rilevante in `vectorium`

- 36d4989 2025-09-10 Merge branch 'main' into develop
    - [Ignorato, modifiche alla documentazione]

- f3aa413 2025-09-10 merge with main
    - [Useless]

- ea1792e 2025-09-10 add index name shrinking
    - [Merged, alcuni conflitti poco significativi su `run_experiments.py`]

- 98ea2cd 2025-09-10 Merge branch 'develop' of https://github.com/rossanoventurini/seismic_private into develop
    - [Skipped, cambiamenti già fatti]

- 05f6569 2025-09-10 uploading experiments configuration file
    - [Merged, solo file di configurazione]

- ab663f4 2025-09-10 added grid search on msmarco v2 cocondenser
    - [cherry-pick --skip, niente di nuovo]

- d6cb1b7 2025-09-10 small cosmetic changes to run_experiments.py
    - [Già presente]

- 0f3f417 2025-09-12 fix problem with fixedu8 and fixedu16
    - [Già presente]

- 8eaed3c 2025-09-16 variable renaming in kmeans
    - [Merged]
    - In occasione del merge è stato risolto il problema della dipendenza da `compressed-intvec`.
    - Ora il Cargo.toml punta a un fork personale.

- 81012f6 2025-09-16 fix error in requirements
    - [Merged, rimosso itertools]

- 4314487 2025-09-17 added best results and best results explanation
    - [Merged, conflitto coi path]

- 3025f32 2025-09-17 adding the notebook for results extraction
    - [cherry-pick --skip, niente di nuovo]

- a1bf936 2025-09-17 Fix typo in README.md regarding configurations
    - [Merged]

- bbb2d1b 2025-09-17 added path to experiments
    - [cherry-pick --skip, niente di nuovo]

- 004e4a7 2025-09-17 Merge remote-tracking branch 'origin/develop'
    - [cherry-pick --skip, niente di nuovo]

- 4cc1395 2025-09-19 Clarify resource usage instructions in README
    - [Merged]

- da332ef 2025-09-19 Change run_experiments command to use 'exp' flag
    - [Merged]

- 9fb1696 2025-09-19 Update run_experiments command and fix typos
    - [Merged]

- 46d3467 2025-09-19 Fix formatting in BestResults.md
    - [Merged]

- 1b60ef4 2025-09-19 Fix warning
    - [Merged]

- 88522e8 2025-09-29 best configs also for esplade and notebook cleaning
    - [Merged]

- dc63ce5 2025-10-06 fixing bincode usage
    - [Merged, conflitti risolti accettando le modifiche in ingresso]

- e0a3326 2025-10-07 fixing logging to console. FM
    - [Merged, ignorate le modifiche al console logging]

- 0118f9d 2025-10-07 Merge branch 'main' of https://github.com/TusKANNy/seismic into main
    - [Ignored, Merge: e0a3326 88522e8]

- 0a8c7d8 2025-10-09 added search function to SeismicDataset
    - [Merged]

- c7d32ef 2025-10-09 Merge branch 'main' of https://github.com/TusKANNy/seismic
    - [Merged]



Il comando 
`git log ecir2026/streamvbyte..main --oneline --no-merges --reverse` 
ci da soltanto i seguenti commit:

- 3256087 Update Guidelines.md
- b9d8ebb Update Guidelines.md
- ea1f31e Update Guidelines.md
- 9281361 Update Guidelines.md
- 5bc746c Update RunExperiments.md
- 2f1327a Update README.md
- 9015083 Various improvements
- 9cf1069 merging Martino's improvements. Wow!
- 2e2e0bd HACK: Faster Fixed point numbers to f32 conversion
- f3aa413 merge with main
- ea1792e add index name shrinking
- 05f6569 uploading experiments configuration file; the grid search to run when adding new feature for cocondenser on msmarcov1 is included
- ab663f4 added grid search on msmarco v2 cocondenser
- d6cb1b7 small cosmetic changes to run_experiments.py
- 0f3f417 fix problem with fixedu8 and fixedu16
- 8eaed3c variable renaming in kmeans
- 81012f6 fix error in requirements
- 4314487 added best results and best results explanation
- 3025f32 adding the notebook for results extraction
- a1bf936 Fix typo in README.md regarding configurations
- bbb2d1b added path to experients
- 4cc1395 Clarify resource usage instructions in README
- da332ef Change run_experiments command to use 'exp' flag
