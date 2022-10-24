mergeInto(LibraryManager.library, {

  UpdateTrialTime: function (t) {
    updateTrialTime(t);
  },

  ChangeTrial: function (trialInc) {
    changeTrial(trialInc);
  },

  TrialViewerLoaded: function() {
    trialViewerLoaded();
  },

  DataLoaded: function() {
    trialViewerDataLoaded();
  }

});