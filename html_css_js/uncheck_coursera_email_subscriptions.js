// just open your developer tool for https://www.coursera.org/account-settings
// then run the following code in browser console

var allInputs = document.getElementsByTagName("input");
for (var i = 0, max = allInputs.length; i < max; i++){
    if (allInputs[i].type === 'checkbox' && allInputs[i].checked === true)
        allInputs[i].click();
}

// finally, manually click `save` buttons to update your account setting
