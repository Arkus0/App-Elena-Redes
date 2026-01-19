import { extractInstagramProfile } from '../instagram-extractor';
import { DomLayoutChangedError } from '../errors';

describe('Instagram Extractor - Semantic Traversal', () => {
  const goldenSampleHtml = `
<header>
  <div class="profile-header-container">
    <div><img alt="Profile picture" src="https://example.com/pic.jpg"></div>

    <section>
      <div>
        <h2>elena_growth_expert</h2>
        <svg aria-label="Verified" role="img" viewBox="0 0 40 40"><path d="..."></path></svg>
      </div>

      <ul>
        <li>
          <span><span class="count">1,250</span> posts</span>
        </li>
        <li>
          <a href="/followers">
            <span title="15,400"><span class="count">15.4k</span> followers</span>
          </a>
        </li>
        <li>
          <a href="/following">
            <span><span class="count">450</span> following</span>
          </a>
        </li>
      </ul>

      <div class="bio-container">
        <span>Elena | Social Media Strategy</span>
        <div>
            🚀 Helping SMBs grow with AI.<br>
            👇 Download my free guide below.<br>
            Madrid, ES.
        </div>
        <a href="https://linktr.ee/elena">linktr.ee/elena</a>
      </div>
    </section>
  </div>
</header>
  `;

  const mockLocation = {
      href: 'https://www.instagram.com/elena_growth_expert/',
      pathname: '/elena_growth_expert/',
  };

  beforeEach(() => {
    document.body.innerHTML = '';
    jest.clearAllMocks();
  });

  it('correctly extracts stats, bio, and verification from Golden Sample', () => {
    document.body.innerHTML = goldenSampleHtml;

    const result = extractInstagramProfile(mockLocation);

    expect(result.success).toBe(true);
    expect(result.profile).toBeDefined();

    const profile = result.profile!;
    expect(profile.username).toBe('elena_growth_expert');
    expect(profile.isVerified).toBe(true);

    // Stats
    expect(profile.stats.posts).toBe(1250);
    expect(profile.stats.followers).toBe(15400); // 15.4k -> 15400
    expect(profile.stats.following).toBe(450);

    // Bio - Should pick the longest text
    expect(profile.bio).toContain('🚀 Helping SMBs grow with AI.');
    expect(profile.bio).toContain('Download my free guide below');
  });

  it('handles Spanish labels correctly', () => {
    const spanishHtml = goldenSampleHtml
      .replace('posts', 'publicaciones')
      .replace('followers', 'seguidores')
      .replace('following', 'seguidos');

    document.body.innerHTML = spanishHtml;

    const result = extractInstagramProfile(mockLocation);

    expect(result.success).toBe(true);
    expect(result.profile?.stats.posts).toBe(1250);
    expect(result.profile?.stats.followers).toBe(15400);
    expect(result.profile?.stats.following).toBe(450);
  });

  it('falls back to rigid selectors if semantic cues are missing', () => {
    // Modify HTML to break semantic cues (remove "followers" text) but keep structure
    const brokenLabelsHtml = goldenSampleHtml
      .replace('posts', '')
      .replace('followers', '')
      .replace('following', '');

    document.body.innerHTML = brokenLabelsHtml;

    const result = extractInstagramProfile(mockLocation);

    expect(result.success).toBe(true);
    // Rigid selectors rely on index 0, 1, 2.
    expect(result.profile?.stats.posts).toBe(1250);
    expect(result.profile?.stats.followers).toBe(15400);
    expect(result.profile?.stats.following).toBe(450);
  });

  it('throws DomLayoutChangedError when structure is completely unrecognizable', () => {
    document.body.innerHTML = '<div><h1>Completely different layout</h1></div>';

    expect(() => {
        extractInstagramProfile(mockLocation);
    }).toThrow(DomLayoutChangedError);
  });
});
