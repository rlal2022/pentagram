import { Container, Card, CardMedia, Grid } from "@mui/material";
import React from "react";

const Gallery: React.FC = () => {
  const images: string[] = Array.from(
    { length: 7 },
    (_, idx) => `/assets/${idx + 1}.jpg`
  );

  return (
    <Container maxWidth="lg" sx={{ mb: 10 }}>
      <Grid container spacing={2}>
        {images.map((image, idx) => (
          <Grid item xs={12} sm={6} md={4} key={idx}>
            <Card sx={{ maxWidth: "100%" }}>
              <CardMedia
                component="img"
                image={image}
                alt={`Image ${idx + 1}`}
                sx={{
                  height: 300,
                  width: "100%",
                  objectFit: "cover",
                }}
              />
            </Card>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
};

export default Gallery;
