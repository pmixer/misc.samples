

// alert("How to degbug");

var tracks = document.getElementsByTagName('track');//$('track');
// alert(tracks.length);
for (var i = 0; i < tracks.length; i++) {
  var t = tracks[i];
  if (t.label == 'Chinese') {
  	var a = document.createElement('a');
  // alert(t.src+"da");
  var href = t.src;
  a.href = href;
  var tokens = href.split("/");
  var len = tokens.len;
  a.download = ('heihei');// TODO: could not name downloaded file, fix this
  a.click();
  }
  
}
