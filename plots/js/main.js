$('.dropdown-menu a').click(function () {
    $('.index #dropdownMenuButton').text($(this).text());
    console.log("Triggered");
});