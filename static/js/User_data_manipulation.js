$(document).ready(function(){
    $('[data-toggle="tooltip"]').tooltip();
    var actions = $("table td:last-child").html();
    $(".add-new").click(function(){
        $(this).attr("disabled", "disabled");
        var index = $("table tbody tr:last-child").index();
        var row = '<tr>' +'<td><input type="date" class="form-control" name="date" id="date"  ></td>'+
            '<td><input type="text" class="form-control" name="Notes" id="text"></td>' +
        '<td>' + actions + '</td>' +
        '</tr>';
        $("table").append(row);  
          $("table tbody tr").eq(index+1).find(".add, .edit").toggle();
        $('[data-toggle="tooltip"]').tooltip();
 
    });
   
    // Add row on add button click
    $(document).on("click", ".add", function(){
        var empty = false;
        var input = $(this).parents("tr").find('input[type="text"]');
        var date = $(this).parents("tr").find('input[type="date"]');
        input.each(function(){
            if(!$(this).val()){
                $(this).addClass("error");
                empty = true;
            } else{
                $(this).removeClass("error");
            }
        });
        date.each(function(){
            if(!$(this).val()){
                $(this).addClass("error");
                empty = true;
                alert("Date value cannot be empty");
            } else{
                $(this).removeClass("error");
            }
        });
        var text = $("#text").val();
        var selectedDate = date.val();
        $.post("/add", { text: text, date:selectedDate}, function(data) {
            $("#displaymessage").html(data);
            $("#displaymessage").show();
        });
        $(this).parents("tr").find(".error").first().focus();
        if(!empty){
            input.each(function(){
                $(this).parent("td").html($(this).val());
            });   
            $(this).parents("tr").find(".add, .edit").toggle();
            $(".add-new").removeAttr("disabled");
        } 
    });
    $(document).on("click", ".delete", function(){
        $(this).parents("tr").remove();
        $(".add-new").removeAttr("disabled");
        var id = $(this).attr("id");
        var string = id;
        $.post("/delete", { string: string}, function(data) {
            $("#displaymessage").html(data);
            $("#displaymessage").show();
        });
    });
$(document).on("click", ".update", function(){
        var id = $(this).attr("id");
        var $row = $(this).closest("tr");
        var string = id;
        var txt = $("#text").val();
        var selectedDat = $row.find('input[name="date"]').val();
        if (txt == "") {
        alert("Enter something in text box");
        }
        if (selectedDat == "") {
            alert("Enter date ");
            }
        else{
        $.post("/edit", { string: string,text: txt,date:selectedDat}, function(data) {
            $("#displaymessage").html(data);
            $("#displaymessage").show();
           
        });}
    });
         // Edit row on edit button click
    $(document).on("click", ".edit", function(){  
        $(this).parents("tr").find("td:not(:last-child)").each(function(i){
            if (i=='1'){
                var idname = 'text';
                var name="Notes"
            }else{
                var idname = 'date';
            var name = "date";
            
            } 
            $(this).html('<input type="'+ idname +'" name="' + name + '" id="' + idname + '" class="form-control" value="' + $(this).text() + '">');
            
                
            
        
        });  
        $(this).parents("tr").find(".add,.update, .edit").toggle();
        $(".add-new").attr("disabled", "disabled");
        $(this).parents("tr").find(".add").removeClass("add").addClass("update"); 
        
         
    });
    
});


// for save button
