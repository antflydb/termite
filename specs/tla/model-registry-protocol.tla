---- MODULE model-registry-protocol ----
EXTENDS Integers, FiniteSets, TLC

\*=========================================================================
\* TLA+ model of the FIXED Termite model registry protocol.
\*
\* Verifies that all 4 bug fixes eliminate the proven violations:
\*   Bug 1 fix: Pre-increment refcount before Get() in Acquire()
\*   Bug 2 fix: All registries have lock+double-check in loadModel()
\*   Bug 4 fix: Eviction callback tracks orphans (no re-add);
\*              Release() closes orphans when refcount hits 0
\*
\* Changes from buggy model:
\*   - Removed HasLoadLock constant (always TRUE now — Bug 2 fix)
\*   - Removed AcquireLoadNoLock action (no longer possible)
\*   - Added AcquirePreIncRef step BEFORE cache lookup (Bug 1 fix)
\*   - Removed "got_handle" state; refcount incremented in "pre_get"
\*   - Replaced CallbackReAdd with CallbackDeferClose (Bug 4 fix)
\*   - Added evictedHandles variable for orphan tracking
\*   - Release() closes orphans when refcount hits 0
\*   - Shutdown closes remaining evicted handles
\*=========================================================================

CONSTANTS
    Models,          \* Set of model names, e.g. {"m1", "m2"}
    Handlers,        \* Set of handler IDs, e.g. {"h1", "h2"}
    MaxHandleId      \* Upper bound on handle IDs for finite model checking

VARIABLES
    \* --- Cache state ---
    cache,           \* Map: model name -> handle ID (or 0 = not cached)
    cacheRunning,    \* Boolean: whether the TTL cleanup goroutine is active

    \* --- Reference counting ---
    refCounts,       \* Map: model name -> integer >= 0

    \* --- Model handle lifecycle ---
    handleState,     \* Map: handle ID -> {"loaded", "closed"}
    nextHandle,      \* Counter: next fresh handle ID to allocate

    \* --- Handler state machine ---
    handlerState,    \* Map: handler -> state in the acquire/use/release protocol
    handlerModel,    \* Map: handler -> model name being acquired (or "")
    handlerHandle,   \* Map: handler -> handle ID obtained (or 0)

    \* --- Loading serialization (always used — Bug 2 fix) ---
    loadLockHeld,    \* Boolean: whether the load mutex is held

    \* --- Pending async eviction callbacks ---
    pendingCallbacks,\* Set of [model |-> name, handle |-> id] awaiting async execution

    \* --- Orphaned handles awaiting cleanup (Bug 4 fix) ---
    evictedHandles,  \* Map: model name -> set of handle IDs tracked for deferred close

    \* --- Shutdown ---
    shutdownPhase    \* "running" | "stopping" | "done"

vars == <<cache, cacheRunning, refCounts, handleState, nextHandle,
          handlerState, handlerModel, handlerHandle, loadLockHeld,
          pendingCallbacks, evictedHandles, shutdownPhase>>

\*=========================================================================
\* HELPER OPERATORS
\*=========================================================================

HandleIds == 1..MaxHandleId

\* Is a handle valid for use?
HandleIsLoaded(h) == h > 0 /\ h \in DOMAIN handleState /\ handleState[h] = "loaded"

\* All handles that exist
AllHandles == {h \in HandleIds : h \in DOMAIN handleState}

\* All evicted handle IDs across all models
AllEvictedHandleIds == UNION {evictedHandles[m] : m \in Models}

\*=========================================================================
\* INITIAL STATE
\*=========================================================================

Init ==
    /\ cache = [m \in Models |-> 0]
    /\ cacheRunning = TRUE
    /\ refCounts = [m \in Models |-> 0]
    /\ handleState = << >>   \* Empty function (no handles yet)
    /\ nextHandle = 1
    /\ handlerState = [h \in Handlers |-> "idle"]
    /\ handlerModel = [h \in Handlers |-> ""]
    /\ handlerHandle = [h \in Handlers |-> 0]
    /\ loadLockHeld = FALSE
    /\ pendingCallbacks = {}
    /\ evictedHandles = [m \in Models |-> {}]
    /\ shutdownPhase = "running"

\*=========================================================================
\* ACTIONS: Handler protocol (Acquire / Use / Release)
\*=========================================================================

\*--- Step 1 of Acquire: pre-increment refcount (Bug 1 fix) ---
\*--- This happens BEFORE Get(), so eviction callback sees refcount > 0 ---
AcquirePreIncRef(h, m) ==
    /\ shutdownPhase = "running"
    /\ handlerState[h] = "idle"
    /\ refCounts' = [refCounts EXCEPT ![m] = @ + 1]
    /\ handlerState' = [handlerState EXCEPT ![h] = "pre_get"]
    /\ handlerModel' = [handlerModel EXCEPT ![h] = m]
    /\ UNCHANGED <<cache, cacheRunning, handleState, nextHandle,
                   handlerHandle, loadLockHeld, pendingCallbacks,
                   evictedHandles, shutdownPhase>>

\*--- Step 2 of Acquire: Get() finds model in cache (cache hit) ---
AcquireCacheHit(h) ==
    /\ handlerState[h] = "pre_get"
    /\ LET m == handlerModel[h] IN
       /\ cache[m] # 0                         \* Model is in cache
       /\ handlerHandle' = [handlerHandle EXCEPT ![h] = cache[m]]
       /\ handlerState' = [handlerState EXCEPT ![h] = "acquired"]
    /\ UNCHANGED <<cache, cacheRunning, refCounts, handleState, nextHandle,
                   handlerModel, loadLockHeld, pendingCallbacks,
                   evictedHandles, shutdownPhase>>

\*--- Step 2 of Acquire: Get() cache miss, acquire load lock ---
\*--- (All registries now have the lock — Bug 2 fix) ---
AcquireLoadLock(h) ==
    /\ handlerState[h] = "pre_get"
    /\ LET m == handlerModel[h] IN
       /\ cache[m] = 0                          \* Cache miss (first check)
       /\ ~loadLockHeld                         \* Acquire lock
       /\ loadLockHeld' = TRUE
       /\ handlerState' = [handlerState EXCEPT ![h] = "loading_locked"]
    /\ UNCHANGED <<cache, cacheRunning, refCounts, handleState, nextHandle,
                   handlerModel, handlerHandle, pendingCallbacks,
                   evictedHandles, shutdownPhase>>

\*--- Double-check after acquiring lock: cache hit (someone else loaded) ---
LoadLockedCacheHit(h) ==
    /\ handlerState[h] = "loading_locked"
    /\ LET m == handlerModel[h] IN
       /\ cache[m] # 0                       \* Someone else loaded it
       /\ handlerHandle' = [handlerHandle EXCEPT ![h] = cache[m]]
       /\ handlerState' = [handlerState EXCEPT ![h] = "acquired"]
       /\ loadLockHeld' = FALSE              \* Release lock
    /\ UNCHANGED <<cache, cacheRunning, refCounts, handleState, nextHandle,
                   handlerModel, pendingCallbacks, evictedHandles, shutdownPhase>>

\*--- Double-check after acquiring lock: still a miss, do the load ---
LoadLockedDoLoad(h) ==
    /\ handlerState[h] = "loading_locked"
    /\ LET m == handlerModel[h] IN
       /\ cache[m] = 0                       \* Still a miss
       /\ nextHandle <= MaxHandleId
       /\ LET newH == nextHandle IN
          /\ handleState' = (newH :> "loaded") @@ handleState
          /\ nextHandle' = nextHandle + 1
          /\ cache' = [cache EXCEPT ![m] = newH]
          /\ handlerHandle' = [handlerHandle EXCEPT ![h] = newH]
          /\ handlerState' = [handlerState EXCEPT ![h] = "acquired"]
          /\ loadLockHeld' = FALSE           \* Release lock
    /\ UNCHANGED <<cacheRunning, refCounts, handlerModel,
                   pendingCallbacks, evictedHandles, shutdownPhase>>

\*--- Handler uses the model ---
UseModel(h) ==
    /\ handlerState[h] = "acquired"
    /\ handlerState' = [handlerState EXCEPT ![h] = "releasing"]
    /\ UNCHANGED <<cache, cacheRunning, refCounts, handleState, nextHandle,
                   handlerModel, handlerHandle, loadLockHeld,
                   pendingCallbacks, evictedHandles, shutdownPhase>>

\*--- Release: decrement refcount, close orphans if count hits 0 (Bug 4 fix) ---
Release(h) ==
    /\ handlerState[h] = "releasing"
    /\ LET m == handlerModel[h] IN
       /\ LET newCount == IF refCounts[m] > 0 THEN refCounts[m] - 1 ELSE 0 IN
          /\ refCounts' = [refCounts EXCEPT ![m] = newCount]
          /\ IF newCount = 0
             THEN \* Close all orphaned handles for this model
                  /\ handleState' = [hId \in DOMAIN handleState |->
                       IF hId \in evictedHandles[m] THEN "closed"
                       ELSE handleState[hId]]
                  /\ evictedHandles' = [evictedHandles EXCEPT ![m] = {}]
             ELSE
                  /\ UNCHANGED <<handleState, evictedHandles>>
       /\ handlerState' = [handlerState EXCEPT ![h] = "idle"]
       /\ handlerModel' = [handlerModel EXCEPT ![h] = ""]
       /\ handlerHandle' = [handlerHandle EXCEPT ![h] = 0]
    /\ UNCHANGED <<cache, cacheRunning, nextHandle,
                   loadLockHeld, pendingCallbacks, shutdownPhase>>

\*=========================================================================
\* ACTIONS: Evictor (non-deterministic TTL/capacity eviction)
\*=========================================================================

\*--- Eviction: remove item from cache and queue async callback ---
Evict(m) ==
    /\ cacheRunning                           \* TTL goroutine is running
    /\ shutdownPhase = "running"
    /\ cache[m] # 0                          \* Model is in cache
    /\ LET h == cache[m] IN
       /\ cache' = [cache EXCEPT ![m] = 0]   \* Remove from cache (sync)
       /\ pendingCallbacks' = pendingCallbacks \cup {[model |-> m, handle |-> h]}
    /\ UNCHANGED <<cacheRunning, refCounts, handleState, nextHandle,
                   handlerState, handlerModel, handlerHandle, loadLockHeld,
                   evictedHandles, shutdownPhase>>

\*--- Async eviction callback: refcount > 0, track as orphan (Bug 4 fix) ---
\*--- DON'T re-add to cache. Track handle for deferred close. ---
CallbackDeferClose(cb) ==
    /\ cb \in pendingCallbacks
    /\ refCounts[cb.model] > 0               \* Model still in use
    /\ evictedHandles' = [evictedHandles EXCEPT ![cb.model] = @ \cup {cb.handle}]
    /\ pendingCallbacks' = pendingCallbacks \ {cb}
    /\ UNCHANGED <<cache, cacheRunning, refCounts, handleState, nextHandle,
                   handlerState, handlerModel, handlerHandle, loadLockHeld,
                   shutdownPhase>>

\*--- Async eviction callback: refcount == 0, close model ---
CallbackClose(cb) ==
    /\ cb \in pendingCallbacks
    /\ refCounts[cb.model] = 0               \* No active references
    /\ cb.handle \in DOMAIN handleState       \* Handle exists
    /\ handleState[cb.handle] = "loaded"      \* Not already closed
    /\ handleState' = [handleState EXCEPT ![cb.handle] = "closed"]
    /\ pendingCallbacks' = pendingCallbacks \ {cb}
    /\ UNCHANGED <<cache, cacheRunning, refCounts, nextHandle,
                   handlerState, handlerModel, handlerHandle, loadLockHeld,
                   evictedHandles, shutdownPhase>>

\*--- Async eviction callback: handle already closed ---
CallbackDoubleClose(cb) ==
    /\ cb \in pendingCallbacks
    /\ refCounts[cb.model] = 0
    /\ cb.handle \in DOMAIN handleState
    /\ handleState[cb.handle] = "closed"      \* Already closed!
    /\ pendingCallbacks' = pendingCallbacks \ {cb}
    /\ UNCHANGED <<cache, cacheRunning, refCounts, handleState, nextHandle,
                   handlerState, handlerModel, handlerHandle, loadLockHeld,
                   evictedHandles, shutdownPhase>>

\*=========================================================================
\* ACTIONS: Shutdown (Close)
\*=========================================================================

\*--- Shutdown step 1: Stop cache (only when all handlers idle) ---
ShutdownStopCache ==
    /\ shutdownPhase = "running"
    /\ \A h \in Handlers : handlerState[h] = "idle"  \* Server drains first
    /\ cacheRunning' = FALSE                  \* No more TTL evictions
    /\ shutdownPhase' = "stopping"
    /\ UNCHANGED <<cache, refCounts, handleState, nextHandle,
                   handlerState, handlerModel, handlerHandle, loadLockHeld,
                   pendingCallbacks, evictedHandles>>

\*--- Shutdown step 2a: Close cached models synchronously ---
ShutdownCloseModel(m) ==
    /\ shutdownPhase = "stopping"
    /\ cache[m] # 0
    /\ LET h == cache[m] IN
       /\ h \in DOMAIN handleState
       /\ handleState[h] = "loaded"
       /\ handleState' = [handleState EXCEPT ![h] = "closed"]
       /\ cache' = [cache EXCEPT ![m] = 0]
    /\ UNCHANGED <<cacheRunning, refCounts, nextHandle,
                   handlerState, handlerModel, handlerHandle, loadLockHeld,
                   pendingCallbacks, evictedHandles, shutdownPhase>>

\*--- Shutdown step 2b: Close remaining evicted handles ---
ShutdownCloseEvicted(m) ==
    /\ shutdownPhase = "stopping"
    /\ evictedHandles[m] # {}
    /\ LET hId == CHOOSE hId \in evictedHandles[m] : TRUE IN
       /\ hId \in DOMAIN handleState
       /\ handleState' = [handleState EXCEPT ![hId] = "closed"]
       /\ evictedHandles' = [evictedHandles EXCEPT ![m] = @ \ {hId}]
    /\ UNCHANGED <<cache, cacheRunning, refCounts, nextHandle,
                   handlerState, handlerModel, handlerHandle, loadLockHeld,
                   pendingCallbacks, shutdownPhase>>

\*--- Shutdown step 3: Done when everything is cleaned up ---
ShutdownComplete ==
    /\ shutdownPhase = "stopping"
    /\ \A m \in Models : cache[m] = 0         \* All cached models closed
    /\ \A m \in Models : evictedHandles[m] = {} \* All orphans closed
    /\ shutdownPhase' = "done"
    /\ UNCHANGED <<cache, cacheRunning, refCounts, handleState, nextHandle,
                   handlerState, handlerModel, handlerHandle, loadLockHeld,
                   pendingCallbacks, evictedHandles>>

\*=========================================================================
\* NEXT-STATE RELATION
\*=========================================================================

Next ==
    \* Handler actions (fixed protocol)
    \/ \E h \in Handlers, m \in Models : AcquirePreIncRef(h, m)
    \/ \E h \in Handlers : AcquireCacheHit(h)
    \/ \E h \in Handlers : AcquireLoadLock(h)
    \/ \E h \in Handlers : LoadLockedCacheHit(h)
    \/ \E h \in Handlers : LoadLockedDoLoad(h)
    \/ \E h \in Handlers : UseModel(h)
    \/ \E h \in Handlers : Release(h)
    \* Evictor actions
    \/ \E m \in Models : Evict(m)
    \* Async callback actions (fixed: no re-add)
    \/ \E cb \in pendingCallbacks : CallbackDeferClose(cb)
    \/ \E cb \in pendingCallbacks : CallbackClose(cb)
    \/ \E cb \in pendingCallbacks : CallbackDoubleClose(cb)
    \* Shutdown actions
    \/ ShutdownStopCache
    \/ \E m \in Models : ShutdownCloseModel(m)
    \/ \E m \in Models : ShutdownCloseEvicted(m)
    \/ ShutdownComplete

\*=========================================================================
\* INVARIANTS
\*=========================================================================

\* Type invariant for basic sanity
TypeInvariant ==
    /\ \A m \in Models : refCounts[m] >= 0
    /\ \A m \in Models : cache[m] \in HandleIds \cup {0}
    /\ \A h \in Handlers : handlerState[h] \in
         {"idle", "pre_get", "loading_locked", "acquired", "releasing"}
    /\ shutdownPhase \in {"running", "stopping", "done"}

\* No handler should ever hold a closed handle.
\* Bug 1 fix ensures refcount is incremented before Get(), so eviction
\* callback sees refcount > 0 and defers close instead of closing.
NoUseAfterClose ==
    \A h \in Handlers :
        (handlerState[h] \in {"acquired", "releasing"} /\ handlerHandle[h] # 0)
        => HandleIsLoaded(handlerHandle[h])

\* No resource leaks — every loaded handle is reachable.
\* A handle is reachable if it's in cache, held by a handler, in a pending
\* callback, OR tracked in evictedHandles (Bug 4 fix adds the last case).
NoResourceLeak ==
    \A hId \in AllHandles :
        handleState[hId] = "loaded" =>
            \/ \E m \in Models : cache[m] = hId
            \/ \E h \in Handlers : handlerHandle[h] = hId
            \/ \E cb \in pendingCallbacks : cb.handle = hId
            \/ hId \in AllEvictedHandleIds

\* No model handle is closed more than once.
NoDoubleClose ==
    \A cb \in pendingCallbacks :
        cb.handle \in DOMAIN handleState =>
            handleState[cb.handle] # "closed"

\* Ref count consistency: refcount matches handlers with active references.
\* With Bug 1 fix, refcount is incremented in pre_get (before Get),
\* so all states from pre_get through releasing count.
RefCountConsistency ==
    \A m \in Models :
        refCounts[m] = Cardinality(
            {h \in Handlers : handlerModel[h] = m
                              /\ handlerState[h] \in
                                 {"pre_get", "loading_locked", "acquired", "releasing"}})

\*=========================================================================
\* SPEC
\*=========================================================================

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

====
