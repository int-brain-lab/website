mergeInto(LibraryManager.library, {

  SelectPID: function (pid) {
    selectPID(UTF8ToString(pid));
  },

  UnityLoaded: function() {
    unityLoaded();
  },

  SelectCluster: function (cluster) {
    selectCluster(cluster);
  },

});