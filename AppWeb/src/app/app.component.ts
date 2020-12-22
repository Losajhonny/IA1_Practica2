import { Component } from '@angular/core';

@Component({
    selector: 'app-root',
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.css']
})
export class AppComponent {
    title = 'AppWeb';

    isFive = false;

    imagenes = [] /*["https://www.wikihow.com/images/2/22/Get-the-URL-for-Pictures-Step-30-Version-2.jpg",
        "https://www.wikihow.com/images/2/22/Get-the-URL-for-Pictures-Step-30-Version-2.jpg",
        "https://www.wikihow.com/images/2/22/Get-the-URL-for-Pictures-Step-30-Version-2.jpg"]*/

    constructor() {
        this.imagenes = [1, 2, 3];
        this.isFive = true;
    }
}
