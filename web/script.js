
// ROAD_CLASS
function ROAD_CLASSList(){
    var values = ['Expressway', 'Major Arterial', 'Other', 'Collector' ,'Minor Arterial', 'Local', 'Pending' ,'Laneway'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "ROAD_CLASS_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
ROAD_CLASSList();

// DISTRICT
function DISTRICTList(){
    var values = ['Toronto and East York' ,'Scarborough' ,'Etobicoke York', 'North York', 'Other'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "DISTRICT_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
DISTRICTList();


// LOCCOORD
function LOCCOORDList(){
    var values = ['Intersection', 'Mid-Block' ,'Exit Ramp', 'Other'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "LOCCOORD_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
LOCCOORDList();

// ACCLOC
function ACCLOCList(){
    var values = ['At Intersection' ,'Non Intersection' ,'At/Near Private Drive', 'Other'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "ACCLOC_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
ACCLOCList();

// TRAFFCTL
function TRAFFCTLList(){
    var values = ['Traffic Signal', 'No Control', 'Pedestrian Crossover', 'Stop Sign', 'Other'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "TRAFFCTL_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
TRAFFCTLList();

// VISIBILITY
function VISIBILITYList(){
    var values = ['Clear', 'Rain', 'Other' ,'Snow', 'Fog, Mist, Smoke, Dust'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "VISIBILITY_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
VISIBILITYList();

// LIGHT
function LIGHTList(){
    var values = ['Daylight', 'Dark', 'Dusk', 'Dawn', 'Other'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "LIGHT_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
LIGHTList();

// RDSFCOND
function RDSFCONDList(){
    var values = ['Dry', 'Wet', 'Slush' , 'Snow' ,'Ice', 'Other'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "RDSFCOND_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
RDSFCONDList();

// IMPACTYPE
function IMPACTYPEList(){
    var values = ['Pedestrian Collisions', 'Turning Movement', 'Angle', 'Approaching', 'SMV Other', 'Rear End',
     'SMV Unattended Vehicle', 'Sideswipe', 'Cyclist Collisions', 'Other'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "IMPACTYPE_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
IMPACTYPEList();

// INVTYPE
function INVTYPEList(){
    var values = ['Driver', 'Pedestrian', 'Passenger' ,'Vehicle Owner', 'Truck Driver', 'Motorcycle Passenger', 'Cyclist Passenger', 'Other'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "INVTYPE_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
INVTYPEList();


// VEHTYPE
function VEHTYPEList(){
    var values = ['Automobile', 'Other', '2 Wheels', 'Bus', 'Truck', 'Van', 'Emergency'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "VEHTYPE_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
VEHTYPEList();


// MOTORCYCLELIST
function MOTORCYCLEList(){
    var values = ['0', '1'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "MOTORCYCLE_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
MOTORCYCLEList();

// CYCLISTLIST
function CYCLISTList(){
    var values = ['0', '1'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "CYCLIST_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
CYCLISTList();

// SPEEDING
function SPEEDINGList(){
    var values = ['0', '1'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "SPEEDING_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
SPEEDINGList();

// REDLIGHTLIST
function REDLIGHTList(){
    var values = ['0', '1'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "REDLIGHT_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
REDLIGHTList();

// DISABILITYLIST
function DISABILITYList(){
    var values = ['0', '1'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "DISABILITY_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
DISABILITYList();

// TRUCKLIST
function TRUCKList(){
    var values = ['0', '1'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "TRUCK_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
TRUCKList();

// TRSN_CITY_VEHLIST
function TRSN_CITY_VEHList(){
    var values = ['0', '1'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "TRSN_CITY_VEH_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
TRSN_CITY_VEHList();

// EMERG_VEHLIST
function EMERG_VEHList(){
    var values = ['0', '1'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "EMERG_VEH_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
EMERG_VEHList();

// EMERG_VEHLIST
function EMERG_VEHList(){
    var values = ['0', '1'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "EMERG_VEH_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
EMERG_VEHList();

// AG_DRIVLIST
function AG_DRIVList(){
    var values = ['0', '1'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "AG_DRIV_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
AG_DRIVList();


// ALCOHOLLIST
function ALCOHOLList(){
    var values = ['0', '1'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "ALCOHOL_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
ALCOHOLList();

// PEDESTRIANLIST
function PEDESTRIANList(){
    var values = ['0', '1'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "PEDESTRIAN_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
PEDESTRIANList();

// PASSENGERLIST
function PASSENGERList(){
    var values = ['0', '1'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "PASSENGER_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
PASSENGERList();


// HOUR
function HOURList(){
var values = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'];
    var dataList = document.createElement('datalist');
    dataList.id = "HOUR_list";

    values.forEach(value =>{
    var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
HOURList();

// AUTOMOBILE
function AUTOMOBILEList(){
    var values = ['0', '1'];

    var dataList = document.createElement('datalist');
    dataList.id = "AUTOMOBILE_list";

    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
AUTOMOBILEList();


// month
function monthList(){
    var values = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'];
    
    var dataList = document.createElement('datalist');
    dataList.id = "month_list";
    
    values.forEach(value =>{
        var option = document.createElement('option');
        option.innerHTML = value;
        option.value = value;
        dataList.appendChild(option);
    })
    document.body.appendChild(dataList);
}
monthList();





// Add active class to the current button (highlight it)
var header = document.getElementById("models");
var btns = header.getElementsByClassName("button");
for (var i = 0; i < btns.length; i++) {
  btns[i].addEventListener("click", function() {
  var current = document.getElementsByClassName("active");
  current[0].className = current[0].className.replace(" active", "");
  this.className += " active";
  });
}


$(document).ready(function () {
    $("#mkp").click(function (e) {
        var modelID = $("#model").val();
        var modelName = "";
        if(modelID == "lr") modelName = "LogisticRegression";
        if(modelID == "rf") modelName = "RandomForestClassifier";
        if(modelID == "dt") modelName = "DecisionTreeClassifier";
        if(modelID == "kn") modelName = "KNeighborsClassifier";
        if(modelID == "svm") modelName = "SVC";
        if(modelID == "hv") modelName = "HardVotingClassifier";
        if(modelID == "sv") modelName = "SoftVotingClassifier";

        var dataJson = JSON.stringify( [{
                            "ROAD_CLASS": $('#ROAD_CLASS').val(),
                            "DISTRICT": $('#DISTRICT').val(),
                            "LOCCOORD": $('#LOCCOORD').val(),
                            "ACCLOC": $('#ACCLOC').val(),
                            "TRAFFCTL": $('#TRAFFCTL').val(),
                            "VISIBILITY": $('#VISIBILITY').val(),
                            "LIGHT": $('#LIGHT').val(),
                            "RDSFCOND": $('#RDSFCOND').val(),
                            "IMPACTYPE": $('#IMPACTYPE').val(),
                            "INVTYPE": $('#INVTYPE').val(),
                            "VEHTYPE": $('#VEHTYPE').val(),
                            "MOTORCYCLE": $('#MOTORCYCLE').val(),
                            "CYCLIST": $('#CYCLIST').val(),
                            "SPEEDING": $('#SPEEDING').val(),
                            "REDLIGHT": $('#REDLIGHT').val(),
                            "DISABILITY": $('#DISABILITY').val(),
                            "TRUCK": $('#TRUCK').val(),
                            "TRSN_CITY_VEH": $('#TRSN_CITY_VEH').val(),
                            "EMERG_VEH": $('#EMERG_VEH').val(),
                            "AG_DRIV": $('#AG_DRIV').val(),
                            "ALCOHOL": $('#ALCOHOL').val(),
                            "PEDESTRIAN": $('#PEDESTRIAN').val(),
                            "PASSENGER": $('#PASSENGER').val(),
                            "HOUR": $('#HOUR').val(),
                            "AUTOMOBILE": $('#AUTOMOBILE').val(),
                            "POLICE_DIVISION": $('#POLICE_DIVISION').val(),
                            "HOOD_ID": $('#HOOD_ID').val(),
                            "month": $('#month').val()
                        }] );

        $.ajax({
            url: "http://localhost:12345/predict/" + modelName,
            dataType: 'json',
            type: 'post',
            contentType: 'application/json',
            data: dataJson,
            processData: false,
            success: function (result,status,xhr) {

                var table = $("<table>");
                table.append("<tr><th>Model</th><td>" + modelName + "</td></tr>");
                table.append("<tr><th>Prediction</th><td>" + result.prediction + "</td></tr>");
                table.append("</table>");
                $("#results").html(table);
            }
        });
    });

    
    $("#scores").click(function (e) {
        var modelID = $("#model").val();
        var modelName = "";
        if(modelID == "lr") modelName = "LogisticRegression";
        if(modelID == "rf") modelName = "RandomForestClassifier";
        if(modelID == "dt") modelName = "DecisionTreeClassifier";
        if(modelID == "kn") modelName = "KNeighborsClassifier";
        if(modelID == "svm") modelName = "SVC";
        if(modelID == "hv") modelName = "HardVotingClassifier";
        if(modelID == "sv") modelName = "SoftVotingClassifier";

        $.ajax({
            url: "http://localhost:12345/scores/" + modelName,
            success: function (result,status,xhr) {

                var table = $("<table>");
                table.append("<tr><th>Model</th><td>" + modelName + "</td></tr>");
                table.append("<tr><th>Accuracy</th><td>" + result.accuracy + "</td></tr>");
                table.append("<tr><th>Precision</th><td>" + result.precision + "</td></tr>");
                table.append("<tr><th>F1</th><td>" + result.f1 + "</td></tr>");
                table.append("<tr><th>Recall</th><td>" + result.recall + "</td></tr>");
                table.append("<tr><th>Roc Auc</th><td>" + result.roc_auc + "</td></tr>");
                table.append("<tr><th>Confussion Matrix</th><td>" + result.confussion_matrix + "</td></tr>");
                table.append("</table>");
                $("#results").html(table);
            }
        });
    });

    

    $("#lr").click(function(){
        $("#model").val("lr");
    });
    $("#rf").click(function(){
        $("#model").val("rf");
    });
    $("#dt").click(function(){
        $("#model").val("dt");
    });
    $("#kn").click(function(){
        $("#model").val("kn");
    });
    $("#svm").click(function(){
        $("#model").val("svm");
    });
    $("#hv").click(function(){
        $("#model").val("hv");
    });
    $("#sv").click(function(){
        $("#model").val("sv");
    });


});