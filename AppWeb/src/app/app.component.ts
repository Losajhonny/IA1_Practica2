import { Component } from '@angular/core';
import { Conexion } from './Conexion'
@Component({
    selector: 'app-root',
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.css']
})
export class AppComponent {
    title = 'AppWeb';
    isFive = false;
    imagenes = [];
    modelos = [{ name: "Usac", exactitud: "80 %" },
                { name: "Mariano", exactitud: "70 %" },
                { name: "Landivar", exactitud: "45 %" },
                { name: "Marroquin", exactitud: "90 %" }];

    constructor() {
        this.isFive = true;
    }

    async subir(target) {
        const files = target.files;
        this.imagenes = []

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const url = await this.toBase64(file)
            this.imagenes.push({ base64: url, name: file.name, prediccion: null })
        }

        this.isFive = this.imagenes.length <= 5 ? true: false;
    }

    toBase64 = file => new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result);
        reader.onerror = error => reject(error);
    })

    async analizar() {
        const info = { images: this.imagenes }
        let res = await Conexion.getInstance().POST("analizar", info)

        this.print(res);

        if (res == null) { return; }

        this.modelos = []
        const porcentajes = res.porcentajes;
        const predicciones = res.predicciones;

        // Actualizar porcentajes de los modelos
        for (let i = 0; i < porcentajes.length; i++)
        {
            const name = porcentajes[i][0]
            const exactitud = Math.round(porcentajes[i][1])
            this.modelos.push({ name: name, exactitud: exactitud + " %" })
        }

        // Actualizar imagenes con la prediccion
        for (let i = 0; i < this.imagenes.length; i++)
        {
            if (predicciones[i][0] == "")
            {
                this.imagenes[i].prediccion = "Ninguno"
            }
            else
            {
                this.imagenes[i].prediccion = predicciones[i][0]
            }
        }
    }

    print(objeto) {
        console.log(objeto)
    }
}
