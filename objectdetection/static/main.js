$(document).ready(function () {

    const dropContainer = document.getElementById('drop-container');
    dropContainer.ondragover = dropContainer.ondragend = function () {
        return false;
    };

    dropContainer.ondrop = function (e) {
        e.preventDefault();
        loadImage(e.dataTransfer.files[0])
    }

    $("#browse-button").change(function () {
        loadImage($("#browse-button").prop("files")[0]);
    });

    $('.modal').modal({
        dismissible: false,
        ready: function (modal, _) {
            $.ajax({
                type: "POST",
                url: '/object_detection/api_request/',
                data: {
                    'image64': $('#img-card-1').attr('src')
                },
                dataType: 'text',
                success: function (data) {
                    loadStats(data)
                }
            }).always(function () {
                modal.modal('close');
            });
        }
    });

    $('#go-back').click(function () {
        $('#img-card-1').removeAttr("src");
        $('#stat-table').html('');
        switchCard(0);
    });
    $('#go-start').click(function () {
        let elem = document.getElementById("result");
        elem.parentNode.removeChild(elem);
        $('#stat-table').html('');
        switchCard(0);
    });

    $('#show').click(function () {
        switchCard(3);
        let timestamp = new Date().getTime();
        let el = document.getElementById("#img-card-2");
        let queryString = "?t=" + timestamp;
        el.src = "http://127.0.0.1:8000/object_detection/Object_Detection/static/test.jpeg" + queryString;
    });

    $('#upload-button').click(function () {
        $('.modal').modal('open');
    });
});

switchCard = function (cardNo) {
    let containers = [".dd-container", ".uf-container", ".dt-container", ".it-container"];
    let visibleContainer = containers[cardNo];
    for (let i = 0; i < containers.length; i++) {
        let oz = (containers[i] === visibleContainer) ? '1' : '0';
        $(containers[i]).animate({
            opacity: oz
        }, {
            duration: 200,
            queue: false,
        }).css("z-index", oz);
    }
}

loadImage = function (file) {
    let reader = new FileReader();
    reader.onload = function (event) {
        $('#img-card-1').attr('src', event.target.result);
    }
    reader.readAsDataURL(file);
    switchCard(1);
}

loadStats = function (jsonData) {
    switchCard(2);
    let data = JSON.parse(jsonData);
    let jtext = data["objects"];
    if (data["success"] === true) {
        let elem = document.createElement("div");
        elem.innerHTML = jsonData;
        elem.setAttribute('id', 'result');
        document.getElementById("result-text").appendChild(elem);
    }
}
