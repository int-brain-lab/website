mergeInto(LibraryManager.library, {

  UpdateTrialTime: function (time) {
    updateTrialTime(time);
  },

  ChangeTrial: function (trialInc) {
    changeTrial(trialInc);
  },

  TrialViewerLoaded: function() {
    trialViewerLoaded();
  }

});