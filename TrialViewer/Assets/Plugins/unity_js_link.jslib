mergeInto(LibraryManager.library, {

  UpdateTrialTime: function (t0, t1, t) {
    updateTrialTime(t0, t1, t);
  },

  ChangeTrial: function (trialInc) {
    changeTrial(trialInc);
  },

  TrialViewerLoaded: function() {
    trialViewerLoaded();
  }

});